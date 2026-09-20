# tune_in_batch.py
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fine-tuning with in-batch negatives using symmetric InfoNCE loss.

For each batch:
    code[i] <-> nl[i]       positive pair
    code[i] <-> nl[j]       j != i: in-batch negative

No CoModel, no PolyLoss, and no momentum encoder are used.
"""

import argparse
import logging
import os
import random
import torch
import json
import numpy as np

from model import Model

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (
    get_linear_schedule_with_warmup,
    RobertaModel,
    RobertaTokenizer
)
from torch.optim import AdamW

import torch.nn.functional as F


logger = logging.getLogger(__name__)


def sim_matrix(a, b, eps=1e-8):
    """
    Cosine similarity matrix.

    Args:
        a: [batch_size, dim]
        b: [batch_size, dim]

    Returns:
        [batch_size, batch_size]
    """
    a_n = a.norm(dim=1)[:, None]
    b_n = b.norm(dim=1)[:, None]

    a_norm = a / torch.max(
        a_n,
        eps * torch.ones_like(a_n)
    )
    b_norm = b / torch.max(
        b_n,
        eps * torch.ones_like(b_n)
    )

    return torch.mm(a_norm, b_norm.transpose(0, 1))


def info_nce_loss(nl_vec, code_vec, temperature=0.05):
    """
    Symmetric InfoNCE loss with in-batch negatives.

    Positive pair:
        nl_vec[i] <-> code_vec[i]

    Negative pairs:
        nl_vec[i] <-> code_vec[j], i != j
    """

    # [batch_size, batch_size]
    logits = sim_matrix(nl_vec, code_vec)

    # Temperature scaling
    logits = logits / temperature

    # Correct pair is on the diagonal.
    labels = torch.arange(
        logits.size(0),
        device=logits.device
    )

    # NL -> Code
    loss = F.cross_entropy(
        logits,
        labels
    )

    return loss

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

    # =========================
    # Original code
    # =========================

    code = (
        " ".join(js["code_tokens"])
        if isinstance(js["code_tokens"], list)
        else " ".join(js["code_tokens"].split())
    )

    code_tokens, code_ids = encode_text(
        code,
        tokenizer,
        args.code_length
    )

    # =========================
    # Original query
    # =========================

    nl = (
        " ".join(js["docstring_tokens"])
        if isinstance(js["docstring_tokens"], list)
        else " ".join(js["doc"].split())
    )

    nl_tokens, nl_ids = encode_text(
        nl,
        tokenizer,
        args.nl_length
    )

    # =========================
    # Generated augmentation
    # =========================

    generated_code_ids = None
    generated_nl_ids = None

    if use_generated:

        generated_code = js["generated_code"]
        generated_query = js["generated_query"]

        _, generated_code_ids = encode_text(
            generated_code,
            tokenizer,
            args.code_length
        )

        _, generated_nl_ids = encode_text(
            generated_query,
            tokenizer,
            args.nl_length
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
                    js = json.loads(line)

                    if 'function_tokens' in js:
                        js['code_tokens'] = js['function_tokens']

                    data.append(js)

            elif "codebase" in file_path or "code_idx_map" in file_path:
                js = json.load(f)

                for key in js:
                    temp = {}
                    temp['code_tokens'] = key.split()
                    temp["retrieval_idx"] = js[key]
                    temp['doc'] = ""
                    temp['docstring_tokens'] = ""
                    data.append(temp)

            elif "json" in file_path:
                for js in json.load(f):
                    data.append(js)

        for js in data:
            self.examples.append(
                convert_examples_to_features(
                    js,
                    tokenizer,
                    args,
                    use_generated
                )
            )

        if "train" in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info(
                    "code_tokens: {}".format(
                        [x.replace('\u0120', '_') for x in example.code_tokens]
                    )
                )
                logger.info(
                    "code_ids: {}".format(
                        ' '.join(map(str, example.code_ids))
                    )
                )
                logger.info(
                    "nl_tokens: {}".format(
                        [x.replace('\u0120', '_') for x in example.nl_tokens]
                    )
                )
                logger.info(
                    "nl_ids: {}".format(
                        ' '.join(map(str, example.nl_ids))
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


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True


def train(args, model, tokenizer):
    """Train the model with in-batch negatives."""

    # get training dataset
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

    # get optimizer and scheduler
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

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d",
        args.train_batch_size // max(args.n_gpu, 1)
    )
    logger.info(
        "  Total train batch size  = %d",
        args.train_batch_size
    )
    logger.info(
        "  Total optimization steps = %d",
        len(train_dataloader) * args.num_train_epochs
    )
    logger.info(
        "  InfoNCE temperature = %f",
        args.temperature
    )

    model.zero_grad()
    model.train()

    tr_num, tr_loss, best_mrr = 0, 0, 0

    checkpoint_prefix = 'checkpoint-best-mrr'
    output_dir = os.path.join(
        args.output_dir,
        '{}'.format(checkpoint_prefix)
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_to_save = model.module if hasattr(model, 'module') else model

    output_dir = os.path.join(
        output_dir,
        '{}'.format('model.bin')
    )

    torch.save(
        model_to_save.state_dict(),
        output_dir
    )

    logger.info(
        "Saving model checkpoint to %s",
        output_dir
    )

    for idx in range(args.num_train_epochs):

        for step, batch in enumerate(train_dataloader):

            # ==========================================================
            # Only original code/NL pairs are used.
            #
            # batch[i][0] = code
            # batch[i][1] = NL
            #
            # Positive:
            #   code[i] <-> nl[i]
            #
            # Negative:
            #   code[i] <-> nl[j], i != j
            # ==========================================================

            code_inputs = batch[0].to(args.device)
            nl_inputs = batch[1].to(args.device)

            # Encode code and NL with the same trainable model
            code_vec = model(
                code_inputs=code_inputs
            )

            nl_vec = model(
                nl_inputs=nl_inputs
            )

            # Symmetric InfoNCE with in-batch negatives
            loss = info_nce_loss(
                nl_vec,
                code_vec,
                temperature=args.temperature
            )

            # report loss
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

            # backward
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm
            )

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

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

        # save best model
        if results['eval_mrr'] > best_mrr:
            best_mrr = results['eval_mrr']

            logger.info("  " + "*" * 20)
            logger.info(
                "  Best mrr:%s",
                round(best_mrr, 4)
            )
            logger.info("  " + "*" * 20)

            checkpoint_prefix = 'checkpoint-best-mrr'

            output_dir = os.path.join(
                args.output_dir,
                '{}'.format(checkpoint_prefix)
            )

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            model_to_save = (
                model.module
                if hasattr(model, 'module')
                else model
            )

            output_dir = os.path.join(
                output_dir,
                '{}'.format('model.bin')
            )

            torch.save(
                model_to_save.state_dict(),
                output_dir
            )

            logger.info(
                "Saving model checkpoint to %s",
                output_dir
            )


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

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info(
        "  Num queries = %d",
        len(query_dataset)
    )
    logger.info(
        "  Num codes = %d",
        len(code_dataset)
    )
    logger.info(
        "  Batch size = %d",
        args.eval_batch_size
    )

    model.eval()

    code_vecs = []
    nl_vecs = []

    for batch in query_dataloader:

        nl_inputs = batch[1].to(args.device)

        with torch.no_grad():
            nl_vec = model(
                nl_inputs=nl_inputs
            )

            nl_vecs.append(
                nl_vec.cpu().numpy()
            )

    for batch in code_dataloader:

        code_inputs = batch[0].to(args.device)

        with torch.no_grad():
            code_vec = model(
                code_inputs=code_inputs
            )

            code_vecs.append(
                code_vec.cpu().numpy()
            )

    model.train()

    code_vecs = np.concatenate(
        code_vecs,
        0
    )

    nl_vecs = np.concatenate(
        nl_vecs,
        0
    )

    scores = np.matmul(
        nl_vecs,
        code_vecs.T
    )

    sort_ids = np.argsort(
        scores,
        axis=-1,
        kind='quicksort',
        order=None
    )[:, ::-1]

    nl_urls = []
    code_urls = []

    for example in query_dataset.examples:
        nl_urls.append(example.url)

    for example in code_dataset.examples:
        code_urls.append(example.url)

    ranks = []

    for url, sort_id in zip(nl_urls, sort_ids):

        rank = 0
        find = False

        for idx in sort_id[:1000]:

            if find is False:
                rank += 1

            if code_urls[idx] == url:
                find = True

        if find:
            ranks.append(1 / rank)
        else:
            ranks.append(0)

    result = {
        "eval_mrr": float(np.mean(ranks))
    }

    return result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--train_data_file",
        default=None,
        type=str,
        help="The input training data file (a json file)."
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written."
    )

    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the MRR(a jsonl file)."
    )

    parser.add_argument(
        "--test_data_file",
        default=None,
        type=str,
        help="An optional input test data file to test the MRR(a josnl file)."
    )

    parser.add_argument(
        "--codebase_file",
        default=None,
        type=str,
        help="An optional input test data file to codebase (a jsonl file)."
    )

    parser.add_argument(
        "--generated_train_data_file",
        default=None,
        type=str
    )

    parser.add_argument(
        "--generated_eval_data_file",
        default=None,
        type=str
    )

    parser.add_argument(
        "--generated_codebase_file",
        default=None,
        type=str
    )

    parser.add_argument(
        "--use_generated",
        action="store_true"
    )

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization."
    )

    parser.add_argument(
        "--frozen_layers",
        default=None,
        type=str,
        help="The layers to freeze during training."
    )

    parser.add_argument(
        "--nl_length",
        default=128,
        type=int,
        help="Optional NL input sequence length after tokenization."
    )

    parser.add_argument(
        "--code_length",
        default=256,
        type=int,
        help="Optional Code input sequence length after tokenization."
    )

    parser.add_argument(
        "--temperature",
        default=0.05,
        type=float,
        help="Temperature used by symmetric InfoNCE loss."
    )

    parser.add_argument(
        "--do_train",
        action='store_true',
        help="Whether to run training."
    )

    parser.add_argument(
        "--do_eval",
        action='store_true',
        help="Whether to run eval on the dev set."
    )

    parser.add_argument(
        "--do_test",
        action='store_true',
        help="Whether to run eval on the test set."
    )

    parser.add_argument(
        "--do_zero_shot",
        action='store_true',
        help="Whether to run eval on the test set."
    )

    parser.add_argument(
        "--do_F2_norm",
        action='store_true',
        help="Whether to run eval on the test set."
    )

    parser.add_argument(
        "--train_batch_size",
        default=4,
        type=int,
        help="Batch size for training."
    )

    parser.add_argument(
        "--eval_batch_size",
        default=4,
        type=int,
        help="Batch size for evaluation."
    )

    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam."
    )

    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm."
    )

    parser.add_argument(
        "--num_train_epochs",
        default=1,
        type=int,
        help="Total number of training epochs to perform."
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="random seed for initialization"
    )

    # print arguments
    args = parser.parse_args()

    print(
        json.dumps(
            vars(args),
            indent=4
        )
    )

    # set log
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )

    # set device
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

    # Set seed
    set_seed(args.seed)

    args.frozen_layers = (
        args.frozen_layers.split(",")
        if args.frozen_layers is not None
        else []
    )

    # build model
    tokenizer = RobertaTokenizer.from_pretrained(
        args.model_name_or_path
    )

    encoder = RobertaModel.from_pretrained(
        args.model_name_or_path
    )

    model = Model(
        encoder,
        args
    )

    logger.info(
        "Training/evaluation parameters %s",
        args
    )

    model.to(args.device)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Training
    if args.do_train:
        train(
            args,
            model,
            tokenizer
        )

    # Evaluation
    if args.do_eval:

        if args.do_zero_shot is False:

            checkpoint_prefix = (
                'checkpoint-best-mrr/model.bin'
            )

            output_dir = os.path.join(
                args.output_dir,
                '{}'.format(checkpoint_prefix)
            )

            model_to_load = (
                model.module
                if hasattr(model, 'module')
                else model
            )

            model_to_load.load_state_dict(
                torch.load(
                    output_dir,
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

    # Test
    if args.do_test:

        if args.do_zero_shot is False:

            checkpoint_prefix = (
                'checkpoint-best-mrr/model.bin'
            )

            output_dir = os.path.join(
                args.output_dir,
                '{}'.format(checkpoint_prefix)
            )

            model_to_load = (
                model.module
                if hasattr(model, 'module')
                else model
            )

            model_to_load.load_state_dict(
                torch.load(
                    output_dir,
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
