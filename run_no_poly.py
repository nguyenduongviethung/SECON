# coding=utf-8
"""
SECON ablation:
- Remove poly_codes
- Remove dot attention
- Keep the Momentum Encoder (EMA update)
- During training, the momentum encoder receives generated code/query
  when --use_generated is enabled.
- The online encoder is used for evaluation exactly as before.
"""

import argparse
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup


logger = logging.getLogger(__name__)


# ============================================================
# Encoder models
# ============================================================

class Model(nn.Module):
    """
    Online encoder.

    This is the encoder used for retrieval/evaluation.
    """

    def __init__(self, encoder, args):
        super().__init__()
        self.encoder = encoder

        for name, param in self.encoder.named_parameters():
            for layer in args.frozen_layers:
                if layer in name:
                    param.requires_grad = False
                    break

    def _encode(self, inputs):
        hidden = self.encoder(
            inputs,
            attention_mask=inputs.ne(1)
        )[0]

        # Same mean pooling as the original Model.
        mask = inputs.ne(1).unsqueeze(-1).float()
        outputs = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)

        return F.normalize(outputs, p=2, dim=1)

    def forward(self, code_inputs=None, nl_inputs=None):
        if code_inputs is not None:
            return self._encode(code_inputs)

        return self._encode(nl_inputs)


class CoModel(nn.Module):
    """
    Momentum encoder without PolyEncoder.

    Removed:
        - poly_code_embeddings
        - dot_attention
        - cross()

    Kept:
        - an independent encoder
        - frozen gradients
        - EMA/Momentum update from the online encoder in train()
    """

    def __init__(self, encoder, args):
        super().__init__()
        self.encoder = encoder

        # Momentum encoder is not optimized by backpropagation.
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _encode(self, inputs):
        hidden = self.encoder(
            inputs,
            attention_mask=inputs.ne(1)
        )[0]

        # No PolyEncoder and no attention:
        # directly use mean-pooled encoder representation.
        mask = inputs.ne(1).unsqueeze(-1).float()
        outputs = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)

        return F.normalize(outputs, p=2, dim=1)

    @torch.no_grad()
    def forward(self, code_inputs=None, nl_inputs=None):
        if code_inputs is not None:
            code_vec = self._encode(code_inputs)
        else:
            code_vec = None

        if nl_inputs is not None:
            nl_vec = self._encode(nl_inputs)
        else:
            nl_vec = None

        return code_vec, nl_vec


# ============================================================
# Loss / utilities
# ============================================================

def covariance_loss(z1: torch.Tensor) -> torch.Tensor:
    N, D = z1.size()

    if N <= 1:
        return torch.zeros([], device=z1.device, dtype=z1.dtype)

    z1 = z1 - z1.mean(dim=0)
    cov_z1 = (z1.T @ z1) / (N - 1)
    diag = torch.eye(D, device=z1.device, dtype=torch.bool)

    return cov_z1[~diag].pow(2).sum() / D


def sim_matrix(a, b, eps=1e-8):
    a_n = a.norm(dim=1)[:, None]
    b_n = b.norm(dim=1)[:, None]

    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))

    return torch.mm(a_norm, b_norm.transpose(0, 1))


def polyloss(view1, view2, margin):
    """
    Keep the original SECON polynomial hard-negative loss.
    This is the LOSS function; it is unrelated to PolyEncoder/poly_codes.
    """
    sim = sim_matrix(view1, view2)
    size = sim.size(0)

    pos = sim.diag()

    eye = torch.eye(size, dtype=torch.bool, device=sim.device)
    neg = sim.masked_fill(eye, -1e9)

    hard_mask = neg + margin > pos.unsqueeze(1)
    hard_neg = neg.masked_fill(~hard_mask, -1e9)

    hardest_neg, _ = hard_neg.max(dim=1)

    valid = (pos < 1 - 1e-5) & (hardest_neg > -1e8)

    pos = pos[valid]
    hardest_neg = hardest_neg[valid]

    if len(pos) == 0:
        return torch.zeros([], device=view1.device, requires_grad=True)

    pos_loss = torch.clamp(
        0.2 * pos.pow(2) - 0.7 * pos + 0.5,
        min=0
    )

    neg_loss = torch.clamp(
        0.9 * hardest_neg.pow(2) - 0.4 * hardest_neg + 0.03,
        min=0
    )

    return (pos_loss + neg_loss)


def loss_fn(nl_vec, code_vec):
    loss1 = polyloss(nl_vec, code_vec, 0.15)
    loss2 = polyloss(code_vec, nl_vec, 0.15)
    return (loss1.sum() + loss2.sum()) / nl_vec.size(0)

def augmentation_alignment_loss(
    nl_vec,
    code_vec,
    generated_nl_vec,
    generated_code_vec,
):
    """
    Align original and generated representations.

    Query alignment:
        query[i] <-> generated_query[i]

    Code alignment:
        code[i] <-> generated_code[i]
    """

    query_loss = loss_fn(
        nl_vec,
        generated_nl_vec
    )

    code_loss = loss_fn(
        code_vec,
        generated_code_vec
    )

    return query_loss + code_loss


# ============================================================
# Dataset
# ============================================================

class InputFeatures(object):
    def __init__(
        self,
        code_tokens,
        code_ids,
        nl_tokens,
        nl_ids,
        url,
        generated_code_ids=None,
        generated_nl_ids=None,
    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url

        self.generated_code_ids = generated_code_ids
        self.generated_nl_ids = generated_nl_ids


def encode_text(text, tokenizer, max_length):
    tokens = tokenizer.tokenize(text)[:max_length - 4]

    tokens = [
        tokenizer.cls_token,
        "<encoder-only>",
        tokenizer.sep_token,
    ] + tokens + [tokenizer.sep_token]

    ids = tokenizer.convert_tokens_to_ids(tokens)

    padding_length = max_length - len(ids)
    ids += [tokenizer.pad_token_id] * padding_length

    return tokens, ids


def convert_examples_to_features(js, tokenizer, args, use_generated=False):
    # Original code
    code = (
        " ".join(js["code_tokens"])
        if isinstance(js["code_tokens"], list)
        else " ".join(js["code_tokens"].split())
    )

    code_tokens, code_ids = encode_text(
        code,
        tokenizer,
        args.code_length,
    )

    # Original query
    nl = (
        " ".join(js["docstring_tokens"])
        if isinstance(js["docstring_tokens"], list)
        else " ".join(js["doc"].split())
    )

    nl_tokens, nl_ids = encode_text(
        nl,
        tokenizer,
        args.nl_length,
    )

    # Generated augmentation
    generated_code_ids = None
    generated_nl_ids = None

    if use_generated:
        generated_code = js["generated_code"]
        generated_query = js["generated_query"]

        _, generated_code_ids = encode_text(
            generated_code,
            tokenizer,
            args.code_length,
        )

        _, generated_nl_ids = encode_text(
            generated_query,
            tokenizer,
            args.nl_length,
        )

    return InputFeatures(
        code_tokens,
        code_ids,
        nl_tokens,
        nl_ids,
        js["url"] if "url" in js else js["retrieval_idx"],
        generated_code_ids,
        generated_nl_ids,
    )


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path, use_generated=False):
        self.args = args
        self.use_generated = use_generated
        self.examples = []

        data = []

        with open(file_path, encoding="utf-8") as f:
            if "jsonl" in file_path:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    js = json.loads(line)

                    if "function_tokens" in js:
                        js["code_tokens"] = js["function_tokens"]

                    data.append(js)

            elif "codebase" in file_path or "code_idx_map" in file_path:
                js = json.load(f)

                for key in js:
                    temp = {
                        "code_tokens": key.split(),
                        "retrieval_idx": js[key],
                        "doc": "",
                        "docstring_tokens": "",
                    }
                    data.append(temp)

            elif "json" in file_path:
                data.extend(json.load(f))

        for js in data:
            self.examples.append(
                convert_examples_to_features(
                    js,
                    tokenizer,
                    args,
                    use_generated=use_generated,
                )
            )

        if "train" in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info(
                    "code_tokens: {}".format(
                        [x.replace("\u0120", "_") for x in example.code_tokens]
                    )
                )
                logger.info(
                    "code_ids: {}".format(
                        " ".join(map(str, example.code_ids))
                    )
                )
                logger.info(
                    "nl_tokens: {}".format(
                        [x.replace("\u0120", "_") for x in example.nl_tokens]
                    )
                )
                logger.info(
                    "nl_ids: {}".format(
                        " ".join(map(str, example.nl_ids))
                    )
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        example = self.examples[i]

        if self.use_generated:
            return (
                torch.tensor(example.code_ids),
                torch.tensor(example.nl_ids),
                torch.tensor(example.generated_code_ids),
                torch.tensor(example.generated_nl_ids),
            )

        return (
            torch.tensor(example.code_ids),
            torch.tensor(example.nl_ids),
        )


# ============================================================
# Training
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True


def train(args, model, cmodel, tokenizer):
    train_file = args.train_data_file

    if args.use_generated:
        train_file = args.generated_train_data_file

    train_dataset = TextDataset(
        tokenizer,
        args,
        train_file,
        use_generated=args.use_generated,
    )

    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=4,
    )

    # Only the online encoder is optimized.
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        eps=1e-8,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * args.num_train_epochs,
    )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info(
        "  Total optimization steps = %d",
        len(train_dataloader) * args.num_train_epochs,
    )

    model.zero_grad()
    cmodel.zero_grad()

    model.train()

    # Momentum encoder must not receive gradients.
    cmodel.eval()

    tr_num, tr_loss, best_mrr = 0, 0, 0

    checkpoint_prefix = "checkpoint-best-mrr"
    output_dir = os.path.join(args.output_dir, checkpoint_prefix)

    os.makedirs(output_dir, exist_ok=True)

    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(
        model_to_save.state_dict(),
        os.path.join(output_dir, "model.bin"),
    )

    logger.info(
        "Saving model checkpoint to %s",
        os.path.join(output_dir, "model.bin"),
    )

    # Stage 0: Align original and generated representations
    #
    # query <-> generated_query
    # code  <-> generated_code
    #
    # This stage is performed before the main query-code training.
    # ==============================================================

    if args.use_generated and args.num_alignment_epochs > 0:

        logger.info("***** Running generated representation alignment *****")
        logger.info(
            "  Num alignment epochs = %d",
            args.num_alignment_epochs
        )

        alignment_optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate,
            eps=1e-8
        )

        alignment_scheduler = get_linear_schedule_with_warmup(
            alignment_optimizer,
            num_warmup_steps=0,
            num_training_steps=(
                len(train_dataloader) *
                args.num_alignment_epochs
            )
        )

        model.train()

        for alignment_epoch in range(
            args.num_alignment_epochs
        ):

            alignment_loss_sum = 0.0

            for step, batch in enumerate(train_dataloader):

                code_inputs = batch[0].to(args.device)
                nl_inputs = batch[1].to(args.device)

                generated_code_inputs = batch[2].to(args.device)
                generated_nl_inputs = batch[3].to(args.device)

                # --------------------------------------------------
                # Encode original and generated query/code
                # --------------------------------------------------

                code_vec = model(
                    code_inputs=code_inputs
                )

                nl_vec = model(
                    nl_inputs=nl_inputs
                )

                generated_code_vec = model(
                    code_inputs=generated_code_inputs
                )

                generated_nl_vec = model(
                    nl_inputs=generated_nl_inputs
                )

                # --------------------------------------------------
                # Alignment loss
                #
                # query <-> generated query
                # code  <-> generated code
                # --------------------------------------------------

                alignment_loss = augmentation_alignment_loss(
                    nl_vec,
                    code_vec,
                    generated_nl_vec,
                    generated_code_vec,
                )

                alignment_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm
                )

                alignment_optimizer.step()
                alignment_optimizer.zero_grad()
                alignment_scheduler.step()

                alignment_loss_sum += alignment_loss.item()

            logger.info(
                "alignment epoch {} loss {}".format(
                    alignment_epoch,
                    round(
                        alignment_loss_sum / len(train_dataloader),
                        5
                    )
                )
            )

            # evaluate
            results = evaluate(
                args,
                model,
                tokenizer,
                args.eval_data_file,
                eval_when_training=True
            )
    
            for key, value in results.items():
                logger.info(
                    "  %s = %s",
                    key,
                    round(value, 4)
                )

        logger.info(
            "***** Finished generated representation alignment *****"
        )

    # ==============================================================
    # Main training optimizer
    # ==============================================================

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        eps=1e-8
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * args.num_train_epochs
    )

    for epoch in range(args.num_train_epochs):

        for step, batch in enumerate(train_dataloader):

            # ------------------------------------------------
            # Original pair -> online encoder
            # ------------------------------------------------
            code_inputs = batch[0].to(args.device)
            nl_inputs = batch[1].to(args.device)

            code_vec = model(code_inputs=code_inputs)
            nl_vec = model(nl_inputs=nl_inputs)

            # ------------------------------------------------
            # Generated pair -> momentum encoder
            # ------------------------------------------------
            if args.use_generated:
                generated_code_inputs = batch[2].to(args.device)
                generated_nl_inputs = batch[3].to(args.device)

                vec1, vec2 = cmodel(
                    code_inputs=generated_code_inputs,
                    nl_inputs=generated_nl_inputs,
                )
            else:
                vec1, vec2 = cmodel(
                    code_inputs=code_inputs,
                    nl_inputs=nl_inputs,
                )

            # ------------------------------------------------
            # Same SECON loss structure
            # ------------------------------------------------
            loss = (
                loss_fn(code_vec, nl_vec)
                + loss_fn(nl_vec, vec1)
                + loss_fn(vec2, code_vec)
                + covariance_loss(code_vec)
            )

            tr_loss += loss.item()
            tr_num += 1

            if (step + 1) % 100 == 0:
                logger.info(
                    "epoch {} step {} loss {}".format(
                        epoch,
                        step + 1,
                        round(tr_loss / tr_num, 5),
                    )
                )
                tr_loss = 0
                tr_num = 0

            # ------------------------------------------------
            # Backprop only through online encoder
            # ------------------------------------------------
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # ------------------------------------------------
            # Momentum update
            #
            # theta_c <- m * theta_c + (1-m) * theta
            # ------------------------------------------------
            with torch.no_grad():
                online_model = (
                    model.module if hasattr(model, "module") else model
                )
                momentum_model = (
                    cmodel.module if hasattr(cmodel, "module") else cmodel
                )

                for p_model, p_cmodel in zip(
                    online_model.parameters(),
                    momentum_model.parameters(),
                ):
                    p_cmodel.data.mul_(args.moco_m).add_(
                        p_model.data,
                        alpha=1.0 - args.moco_m,
                    )

        # ----------------------------------------------------
        # Evaluate online encoder only
        # ----------------------------------------------------
        results = evaluate(
            args,
            model,
            tokenizer,
            args.eval_data_file,
            eval_when_training=True,
        )

        for key, value in results.items():
            logger.info("  %s = %s", key, round(value, 4))

        if results["eval_mrr"] > best_mrr:
            best_mrr = results["eval_mrr"]

            logger.info("********************")
            logger.info("Best mrr:%s", round(best_mrr, 4))
            logger.info("********************")

            checkpoint_prefix = "checkpoint-best-mrr"
            output_dir = os.path.join(args.output_dir, checkpoint_prefix)
            os.makedirs(output_dir, exist_ok=True)

            model_to_save = (
                model.module if hasattr(model, "module") else model
            )

            output_path = os.path.join(output_dir, "model.bin")

            torch.save(
                model_to_save.state_dict(),
                output_path,
            )

            logger.info("Saving model checkpoint to %s", output_path)


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    args,
    model,
    tokenizer,
    file_name,
    eval_when_training=False,
):
    query_dataset = TextDataset(
        tokenizer,
        args,
        file_name,
        use_generated=False,
    )

    query_sampler = SequentialSampler(query_dataset)

    query_dataloader = DataLoader(
        query_dataset,
        sampler=query_sampler,
        batch_size=args.eval_batch_size,
        num_workers=4,
    )

    code_dataset = TextDataset(
        tokenizer,
        args,
        args.codebase_file,
        use_generated=False,
    )

    code_sampler = SequentialSampler(code_dataset)

    code_dataloader = DataLoader(
        code_dataset,
        sampler=code_sampler,
        batch_size=args.eval_batch_size,
        num_workers=4,
    )

    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()

    code_vecs = []
    nl_vecs = []

    for batch in query_dataloader:
        nl_inputs = batch[1].to(args.device)

        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs)
            nl_vecs.append(nl_vec.cpu().numpy())

    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)

        with torch.no_grad():
            code_vec = model(code_inputs=code_inputs)
            code_vecs.append(code_vec.cpu().numpy())

    model.train()

    code_vecs = np.concatenate(code_vecs, 0)
    nl_vecs = np.concatenate(nl_vecs, 0)

    scores = np.matmul(nl_vecs, code_vecs.T)

    sort_ids = np.argsort(
        scores,
        axis=-1,
        kind="quicksort",
    )[:, ::-1]

    nl_urls = [example.url for example in query_dataset.examples]
    code_urls = [example.url for example in code_dataset.examples]

    ranks = []

    for url, sort_id in zip(nl_urls, sort_ids):
        rank = 0
        find = False

        for idx in sort_id[:1000]:
            if not find:
                rank += 1

            if code_urls[idx] == url:
                find = True

        if find:
            ranks.append(1 / rank)
        else:
            ranks.append(0)

    return {
        "eval_mrr": float(np.mean(ranks)),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file",
        default=None,
        type=str,
        help="The input training data file.",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory.",
    )

    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="Evaluation data file.",
    )

    parser.add_argument(
        "--test_data_file",
        default=None,
        type=str,
        help="Test data file.",
    )

    parser.add_argument(
        "--codebase_file",
        default=None,
        type=str,
        help="Codebase file.",
    )

    # Generated augmentation
    parser.add_argument(
        "--generated_train_data_file",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--use_generated",
        action="store_true",
        help="Use generated query/code as inputs to the momentum encoder.",
    )

    # Model
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization.",
    )

    parser.add_argument(
        "--frozen_layers",
        default=None,
        type=str,
        help="Layers to freeze in the online encoder.",
    )

    parser.add_argument(
        "--moco_m",
        default=0.999,
        type=float,
        help="Momentum coefficient for the key encoder.",
    )

    # Sequence lengths
    parser.add_argument(
        "--nl_length",
        default=128,
        type=int,
    )

    parser.add_argument(
        "--code_length",
        default=256,
        type=int,
    )

    # Modes
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--do_zero_shot", action="store_true")
    parser.add_argument("--do_F2_norm", action="store_true")

    # Training
    parser.add_argument(
        "--train_batch_size",
        default=4,
        type=int,
    )

    parser.add_argument(
        "--eval_batch_size",
        default=4,
        type=int,
    )

    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
    )

    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
    )

    parser.add_argument(
        "--num_train_epochs",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--num_alignment_epochs",
        default=0,
        type=int,
        help="Number of epochs for aligning original and generated "
            "query/code representations before main training."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    print(json.dumps(vars(args), indent=4))

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logger.info(
        "device: %s, n_gpu: %s",
        device,
        args.n_gpu,
    )

    set_seed(args.seed)

    args.frozen_layers = (
        args.frozen_layers.split(",")
        if args.frozen_layers is not None
        else []
    )

    # --------------------------------------------------------
    # Build online encoder and momentum encoder
    # --------------------------------------------------------
    tokenizer = RobertaTokenizer.from_pretrained(
        args.model_name_or_path
    )

    model = RobertaModel.from_pretrained(
        args.model_name_or_path
    )

    model2 = RobertaModel.from_pretrained(
        args.model_name_or_path
    )

    model = Model(model, args)
    cmodel = CoModel(model2, args)

    logger.info(
        "Training/evaluation parameters %s",
        args,
    )

    model.to(args.device)
    cmodel.to(args.device)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        cmodel = torch.nn.DataParallel(cmodel)

    # Training
    if args.do_train:
        train(
            args,
            model,
            cmodel,
            tokenizer,
        )

    # Evaluation
    if args.do_eval:
        if args.do_zero_shot is False:
            checkpoint_prefix = "checkpoint-best-mrr/model.bin"
            output_dir = os.path.join(
                args.output_dir,
                checkpoint_prefix,
            )

            model_to_load = (
                model.module
                if hasattr(model, "module")
                else model
            )

            model_to_load.load_state_dict(
                torch.load(
                    output_dir,
                    map_location=args.device,
                )
            )

        model.to(args.device)

        result = evaluate(
            args,
            model,
            tokenizer,
            args.eval_data_file,
        )

        logger.info("***** Eval results *****")

        for key in sorted(result.keys()):
            logger.info(
                "  %s = %s",
                key,
                str(round(result[key], 3)),
            )

    # Test
    if args.do_test:
        if args.do_zero_shot is False:
            checkpoint_prefix = "checkpoint-best-mrr/model.bin"
            output_dir = os.path.join(
                args.output_dir,
                checkpoint_prefix,
            )

            model_to_load = (
                model.module
                if hasattr(model, "module")
                else model
            )

            model_to_load.load_state_dict(
                torch.load(
                    output_dir,
                    map_location=args.device,
                )
            )

        model.to(args.device)

        result = evaluate(
            args,
            model,
            tokenizer,
            args.test_data_file,
        )

        logger.info("***** Test results *****")

        for key in sorted(result.keys()):
            logger.info(
                "  %s = %s",
                key,
                str(round(result[key], 3)),
            )


if __name__ == "__main__":
    main()
