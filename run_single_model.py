# coding=utf-8
"""
SECON variant without Momentum Encoder (CoModel).

Main changes from the original run.py:
1. Only ONE encoder/model is used.
2. The same encoder creates original and augmented representations.
3. Poly-code augmentation is performed during training.
4. Evaluation uses only the normal query/code embedding (no poly augmentation).
5. No moco_m, CoModel, second optimizer, or EMA update.
"""

import argparse
import logging
import os
import random
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import get_linear_schedule_with_warmup, RobertaModel, RobertaTokenizer
from torch.optim import AdamW

logger = logging.getLogger(__name__)


# ============================================================
# Loss / similarity
# ============================================================

def covariance_loss(z1: torch.Tensor) -> torch.Tensor:
    N, D = z1.size()

    # Avoid division by zero for very small batches.
    if N <= 1:
        return torch.zeros([], device=z1.device, dtype=z1.dtype)

    z1 = z1 - z1.mean(dim=0)
    cov_z1 = (z1.T @ z1) / (N - 1)

    diag = torch.eye(D, device=z1.device, dtype=torch.bool)
    return cov_z1.masked_fill(diag, 0).pow(2).sum() / D


def sim_matrix(a, b, eps=1e-8):
    a_n = a.norm(dim=1)[:, None]
    b_n = b.norm(dim=1)[:, None]

    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)

    return torch.mm(a_norm, b_norm.transpose(0, 1))


def polyloss(view1, view2, margin):
    """
    Same polynomial hard-negative loss used by the supplied run.py,
    implemented with tensor operations.
    """
    sim = sim_matrix(view1, view2)
    size = sim.size(0)

    if size <= 1:
        return torch.zeros([], device=sim.device, dtype=sim.dtype)

    pos = sim.diag()

    eye = torch.eye(size, dtype=torch.bool, device=sim.device)
    neg = sim.masked_fill(eye, -1e9)

    hard_mask = neg + margin > pos.unsqueeze(1)
    hard_neg = neg.masked_fill(~hard_mask, -1e9)

    hardest_neg, _ = hard_neg.max(dim=1)

    valid = (pos < 1 - 1e-5) & (hardest_neg > -1e8)

    if not valid.any():
        # Keep a gradient path to the model.
        return sim.sum() * 0.0

    pos = pos[valid]
    hardest_neg = hardest_neg[valid]

    pos_loss = torch.clamp(
        0.2 * pos.pow(2) - 0.7 * pos + 0.5,
        min=0
    )

    neg_loss = torch.clamp(
        0.9 * hardest_neg.pow(2) - 0.4 * hardest_neg + 0.03,
        min=0
    )

    return (pos_loss + neg_loss).sum()


def loss_fn(nl_vec, code_vec):
    loss1 = polyloss(nl_vec, code_vec, 0.15)
    loss2 = polyloss(code_vec, nl_vec, 0.15)

    return (loss1 + loss2) / nl_vec.size(0)


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
        tokenizer.sep_token
    ] + tokens + [tokenizer.sep_token]

    ids = tokenizer.convert_tokens_to_ids(tokens)

    padding_length = max_length - len(ids)
    ids += [tokenizer.pad_token_id] * padding_length

    return tokens, ids


def convert_examples_to_features(js, tokenizer, args, use_generated=False):
    code = (
        " ".join(js["code_tokens"])
        if isinstance(js["code_tokens"], list)
        else " ".join(js["code_tokens"].split())
    )

    code_tokens, code_ids = encode_text(
        code, tokenizer, args.code_length
    )

    nl = (
        " ".join(js["docstring_tokens"])
        if isinstance(js["docstring_tokens"], list)
        else " ".join(js["doc"].split())
    )

    nl_tokens, nl_ids = encode_text(
        nl, tokenizer, args.nl_length
    )

    generated_code_ids = None
    generated_nl_ids = None

    if use_generated:
        generated_code = js["generated_code"]
        generated_query = js["generated_query"]

        _, generated_code_ids = encode_text(
            generated_code, tokenizer, args.code_length
        )

        _, generated_nl_ids = encode_text(
            generated_query, tokenizer, args.nl_length
        )

    return InputFeatures(
        code_tokens,
        code_ids,
        nl_tokens,
        nl_ids,
        js["url"] if "url" in js else js["retrieval_idx"],
        generated_code_ids,
        generated_nl_ids
    )


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path, use_generated=False):
        self.args = args
        self.use_generated = use_generated
        self.examples = []

        data = []
        with open(file_path) as f:
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
                        "docstring_tokens": ""
                    }
                    data.append(temp)

            elif "json" in file_path:
                data.extend(json.load(f))

        for js in data:
            self.examples.append(
                convert_examples_to_features(
                    js, tokenizer, args, use_generated
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
            torch.tensor(example.nl_ids)
        )


# ============================================================
# Single-model SECON
# ============================================================

class SingleModel(nn.Module):
    """
    One shared encoder + learnable poly codes.

    normal embedding:
        input -> encoder -> mean pooling -> L2 normalization

    augmented embedding:
        input -> encoder hidden states
              -> poly-code attention
              -> candidate attention
              -> L2 normalization

    During training:
        original q/c are encoded normally.
        generated q/c (when --use_generated is enabled) are encoded by
        the SAME encoder and passed through poly augmentation.

    During evaluation:
        only the normal embedding path is used.
    """

    def __init__(self, encoder, args):
        super().__init__()

        self.encoder = encoder
        self.args = args

        self.poly_m = args.poly_m
        self.poly_code_dim = args.poly_code_dim

        # RoBERTa hidden size should normally equal poly_code_dim.
        hidden_size = self.encoder.config.hidden_size
        if hidden_size != self.poly_code_dim:
            raise ValueError(
                "poly_code_dim ({}) must equal encoder hidden_size ({}) "
                "for the current SECON poly-attention implementation."
                .format(self.poly_code_dim, hidden_size)
            )

        self.poly_code_embeddings = nn.Embedding(
            self.poly_m,
            self.poly_code_dim
        )

        # Keep the original initialization behavior of the supplied CoModel:
        # PyTorch Embedding initialization is used by default.

        for name, param in self.encoder.named_parameters():
            for ele in args.frozen_layers:
                if ele in name:
                    param.requires_grad = False
                    break

    def mean_pool(self, hidden, inputs):
        mask = inputs.ne(1).unsqueeze(-1).type_as(hidden)

        denom = mask.sum(dim=1).clamp(min=1.0)

        outputs = (hidden * mask).sum(dim=1) / denom

        return F.normalize(outputs, p=2, dim=1)

    def encode_hidden(self, inputs):
        return self.encoder(
            inputs,
            attention_mask=inputs.ne(1)
        )[0]

    def encode_normal(self, inputs):
        hidden = self.encode_hidden(inputs)
        return self.mean_pool(hidden, inputs)

    def dot_attention(self, q, k, v):
        # q: [bs, q_len, dim]
        # k/v: [bs, seq_len, dim]
        attn_weights = torch.matmul(q, k.transpose(2, 1))
        attn_weights = F.softmax(attn_weights, dim=-1)

        return torch.matmul(attn_weights, v)

    def poly_augment(self, ctx_hidden, cand_hidden):
        """
        Construct one query/code augmented embedding using the same
        encoder and the learnable poly codes.

        This follows the direction of the original CoModel:
            poly codes -> attend to context hidden states
            candidate CLS -> attend to poly representations
        """
        bs = ctx_hidden.size(0)

        poly_code_ids = torch.arange(
            self.poly_m,
            device=ctx_hidden.device
        ).unsqueeze(0).expand(bs, self.poly_m)

        poly_codes = self.poly_code_embeddings(poly_code_ids)

        # [bs, poly_m, hidden]
        embs = self.dot_attention(
            poly_codes,
            ctx_hidden,
            ctx_hidden
        )

        # Candidate CLS representation
        cand_emb = cand_hidden[:, 0, :].unsqueeze(1)

        # [bs, 1, hidden]
        augmented = self.dot_attention(
            cand_emb,
            embs,
            embs
        )

        return F.normalize(augmented[:, 0, :], p=2, dim=1)

    def forward(
        self,
        code_inputs=None,
        nl_inputs=None,
        generated_code_inputs=None,
        generated_nl_inputs=None,
        augment=False
    ):
        """
        Returns:
            normal_code, normal_nl, augmented_code, augmented_nl

        For inference, call with only code_inputs/nl_inputs and augment=False.
        """

        normal_code = None
        normal_nl = None
        augmented_code = None
        augmented_nl = None

        # --------------------------------------------------------
        # Normal representations
        # --------------------------------------------------------
        code_hidden = None
        nl_hidden = None

        if code_inputs is not None:
            code_hidden = self.encode_hidden(code_inputs)
            normal_code = self.mean_pool(code_hidden, code_inputs)

        if nl_inputs is not None:
            nl_hidden = self.encode_hidden(nl_inputs)
            normal_nl = self.mean_pool(nl_hidden, nl_inputs)

        # --------------------------------------------------------
        # Augmented representations
        # --------------------------------------------------------
        if augment:
            # If generated inputs are not supplied, use the original
            # inputs. This makes the model usable without generated data.
            if generated_code_inputs is None:
                generated_code_inputs = code_inputs

            if generated_nl_inputs is None:
                generated_nl_inputs = nl_inputs

            generated_code_hidden = self.encode_hidden(
                generated_code_inputs
            )
            generated_nl_hidden = self.encode_hidden(
                generated_nl_inputs
            )

            # Same direction as the original CoModel:
            #
            # v1 = cross(code_hidden, nl_hidden)
            #      -> augmented query
            #
            # v2 = cross(nl_hidden, code_hidden)
            #      -> augmented code
            #
            # Therefore:
            #   original query  <-> augmented query
            #   augmented code  <-> original code
            augmented_nl = self.poly_augment(
                generated_code_hidden,
                generated_nl_hidden
            )

            augmented_code = self.poly_augment(
                generated_nl_hidden,
                generated_code_hidden
            )

        return normal_code, normal_nl, augmented_code, augmented_nl


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


def train(args, model, tokenizer):
    train_file = args.train_data_file

    if args.use_generated:
        train_file = args.generated_train_data_file

    train_dataset = TextDataset(
        tokenizer,
        args,
        train_file,
        use_generated=args.use_generated
    )

    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=4
    )

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

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d",
        args.train_batch_size // max(args.n_gpu, 1)
    )
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info(
        "  Total optimization steps = %d",
        len(train_dataloader) * args.num_train_epochs
    )

    model.zero_grad()

    tr_num = 0
    tr_loss = 0
    best_mrr = 0

    checkpoint_dir = os.path.join(
        args.output_dir,
        "checkpoint-best-mrr"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    for idx in range(args.num_train_epochs):
        model.train()

        for step, batch in enumerate(train_dataloader):
            code_inputs = batch[0].to(args.device)
            nl_inputs = batch[1].to(args.device)

            generated_code_inputs = None
            generated_nl_inputs = None

            if args.use_generated:
                generated_code_inputs = batch[2].to(args.device)
                generated_nl_inputs = batch[3].to(args.device)

            (
                code_vec,
                nl_vec,
                aug_code_vec,
                aug_nl_vec
            ) = model(
                code_inputs=code_inputs,
                nl_inputs=nl_inputs,
                generated_code_inputs=generated_code_inputs,
                generated_nl_inputs=generated_nl_inputs,
                augment=True
            )

            # Same semantic structure as the original run.py:
            #
            # original code <-> original query
            # original query <-> augmented query
            # augmented code <-> original code
            #
            # but ALL representations are now produced by ONE encoder.
            loss = (
                loss_fn(code_vec, nl_vec)
                + loss_fn(nl_vec, aug_nl_vec)
                + loss_fn(aug_code_vec, code_vec)
                + covariance_loss(code_vec)
            )

            tr_loss += loss.item()
            tr_num += 1

            if (step + 1) % 100 == 0:
                logger.info(
                    "epoch {} step {} loss {}".format(
                        idx,
                        step + 1,
                        round(tr_loss / tr_num, 5)
                    )
                )
                tr_loss = 0
                tr_num = 0

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm
            )

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # Evaluate with NORMAL embeddings only.
        results = evaluate(
            args,
            model,
            tokenizer,
            args.eval_data_file,
            eval_when_training=True
        )

        for key, value in results.items():
            logger.info("  %s = %s", key, round(value, 4))

        if results["eval_mrr"] > best_mrr:
            best_mrr = results["eval_mrr"]

            logger.info("********************")
            logger.info("Best mrr:%s", round(best_mrr, 4))
            logger.info("********************")

            model_to_save = (
                model.module if hasattr(model, "module") else model
            )

            output_file = os.path.join(
                checkpoint_dir,
                "model.bin"
            )

            torch.save(
                model_to_save.state_dict(),
                output_file
            )

            logger.info(
                "Saving model checkpoint to %s",
                output_file
            )


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    args,
    model,
    tokenizer,
    file_name,
    eval_when_training=False
):
    query_dataset = TextDataset(
        tokenizer,
        args,
        file_name
    )

    query_sampler = SequentialSampler(query_dataset)

    query_dataloader = DataLoader(
        query_dataset,
        sampler=query_sampler,
        batch_size=args.eval_batch_size,
        num_workers=4
    )

    code_dataset = TextDataset(
        tokenizer,
        args,
        args.codebase_file
    )

    code_sampler = SequentialSampler(code_dataset)

    code_dataloader = DataLoader(
        code_dataset,
        sampler=code_sampler,
        batch_size=args.eval_batch_size,
        num_workers=4
    )

    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()

    code_vecs = []
    nl_vecs = []

    # IMPORTANT:
    # Evaluation never uses poly augmentation.
    # It is a standard bi-encoder representation.
    with torch.no_grad():
        for batch in query_dataloader:
            nl_inputs = batch[1].to(args.device)

            _, nl_vec, _, _ = model(
                nl_inputs=nl_inputs,
                augment=False
            )

            nl_vecs.append(
                nl_vec.cpu().numpy()
            )

        for batch in code_dataloader:
            code_inputs = batch[0].to(args.device)

            code_vec, _, _, _ = model(
                code_inputs=code_inputs,
                augment=False
            )

            code_vecs.append(
                code_vec.cpu().numpy()
            )

    if not eval_when_training:
        model.train()

    code_vecs = np.concatenate(code_vecs, 0)
    nl_vecs = np.concatenate(nl_vecs, 0)

    scores = np.matmul(nl_vecs, code_vecs.T)

    sort_ids = np.argsort(
        scores,
        axis=-1,
        kind="quicksort"
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
                break

        if find:
            ranks.append(1 / rank)
        else:
            ranks.append(0)

    return {
        "eval_mrr": float(np.mean(ranks))
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    parser.add_argument(
        "--train_data_file",
        default=None,
        type=str,
        help="The input training data file."
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="Output directory."
    )

    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="Evaluation data file."
    )

    parser.add_argument(
        "--test_data_file",
        default=None,
        type=str,
        help="Test data file."
    )

    parser.add_argument(
        "--codebase_file",
        default=None,
        type=str,
        help="Codebase file."
    )

    parser.add_argument(
        "--generated_train_data_file",
        default=None,
        type=str
    )

    parser.add_argument(
        "--use_generated",
        action="store_true",
        help="Use LLM-generated query/code as the augmented views."
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization."
    )

    parser.add_argument(
        "--poly_m",
        default=16,
        type=int,
        help="Number of learnable poly codes."
    )

    parser.add_argument(
        "--poly_code_dim",
        default=768,
        type=int,
        help="Dimension of poly codes; must equal encoder hidden size."
    )

    parser.add_argument(
        "--frozen_layers",
        default=None,
        type=str,
        help="Comma-separated encoder layers to freeze."
    )

    parser.add_argument(
        "--nl_length",
        default=128,
        type=int,
        help="NL sequence length."
    )

    parser.add_argument(
        "--code_length",
        default=256,
        type=int,
        help="Code sequence length."
    )

    # --------------------------------------------------------
    # Mode
    # --------------------------------------------------------
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Whether to run training."
    )

    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Whether to evaluate on dev."
    )

    parser.add_argument(
        "--do_test",
        action="store_true",
        help="Whether to evaluate on test."
    )

    parser.add_argument(
        "--do_zero_shot",
        action="store_true",
        help="Skip checkpoint loading."
    )

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------
    parser.add_argument(
        "--train_batch_size",
        default=4,
        type=int
    )

    parser.add_argument(
        "--eval_batch_size",
        default=4,
        type=int
    )

    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float
    )

    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float
    )

    parser.add_argument(
        "--num_train_epochs",
        default=1,
        type=int
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    print(json.dumps(vars(args), indent=4))

    logging.basicConfig(
        format=(
            "%(asctime)s - %(levelname)s - %(name)s - "
            "%(message)s"
        ),
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logger.info(
        "device: %s, n_gpu: %s",
        device,
        args.n_gpu
    )

    set_seed(args.seed)

    args.frozen_layers = (
        args.frozen_layers.split(",")
        if args.frozen_layers is not None
        else []
    )

    # --------------------------------------------------------
    # Build ONE model only
    # --------------------------------------------------------
    tokenizer = RobertaTokenizer.from_pretrained(
        args.model_name_or_path
    )

    encoder = RobertaModel.from_pretrained(
        args.model_name_or_path
    )

    model = SingleModel(
        encoder,
        args
    )

    model.to(args.device)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info(
        "Training/evaluation parameters %s",
        args
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    if args.do_train:
        train(
            args,
            model,
            tokenizer
        )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    if args.do_eval:
        if not args.do_zero_shot:
            checkpoint_prefix = "checkpoint-best-mrr/model.bin"
            output_file = os.path.join(
                args.output_dir,
                checkpoint_prefix
            )

            model_to_load = (
                model.module
                if hasattr(model, "module")
                else model
            )

            model_to_load.load_state_dict(
                torch.load(
                    output_file,
                    map_location=args.device
                )
            )

        model.to(args.device)

        result = evaluate(
            args,
            model,
            tokenizer,
            args.eval_data_file
        )

        logger.info("***** Eval results *****")

        for key in sorted(result.keys()):
            logger.info(
                "  %s = %s",
                key,
                str(round(result[key], 3))
            )

    if args.do_test:
        if not args.do_zero_shot:
            checkpoint_prefix = "checkpoint-best-mrr/model.bin"
            output_file = os.path.join(
                args.output_dir,
                checkpoint_prefix
            )

            model_to_load = (
                model.module
                if hasattr(model, "module")
                else model
            )

            model_to_load.load_state_dict(
                torch.load(
                    output_file,
                    map_location=args.device
                )
            )

        model.to(args.device)

        result = evaluate(
            args,
            model,
            tokenizer,
            args.test_data_file
        )

        logger.info("***** Test results *****")

        for key in sorted(result.keys()):
            logger.info(
                "  %s = %s",
                key,
                str(round(result[key], 3))
            )


if __name__ == "__main__":
    main()
