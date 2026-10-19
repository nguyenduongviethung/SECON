# coding=utf-8
"""Fine-tune a dual encoder to retrieve code summaries from queries."""

import argparse
import csv
import json
import logging
import os

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup

from run_encoder import (
    Model,
    covariance_loss,
    loss_fn,
    save_best_model,
    set_seed,
)


logger = logging.getLogger(__name__)


class SummaryFeatures:
    def __init__(self, query_tokens, query_ids, summary_tokens, summary_ids, url):
        self.query_tokens = query_tokens
        self.query_ids = query_ids
        self.summary_tokens = summary_tokens
        self.summary_ids = summary_ids
        self.url = url


def encode_text(text, tokenizer, max_length):
    tokens = tokenizer.tokenize(text)[:max_length - 4]
    tokens = [
        tokenizer.cls_token,
        "<encoder-only>",
        tokenizer.sep_token,
    ] + tokens + [tokenizer.sep_token]
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids += [tokenizer.pad_token_id] * (max_length - len(ids))
    return tokens, ids


def _record_url(record):
    return record["url"] if "url" in record else record["retrieval_idx"]


class SummaryDataset(Dataset):
    """Query-summary pairs, with the text field selected by dataset role."""

    def __init__(self, tokenizer, args, file_path, text_field):
        self.examples = []

        with open(file_path, encoding="utf-8") as data_file:
            if "jsonl" in file_path:
                records = [
                    json.loads(line)
                    for line in data_file
                    if line.strip()
                ]
            elif "json" in file_path:
                records = json.load(data_file)
            else:
                raise ValueError(
                    "Dataset must be a JSON or JSONL file: {}".format(file_path)
                )

        for index, record in enumerate(records):
            query_text = record.get("docstring", "")
            summary_text = record.get("summary", "")
            if text_field == "pair":
                missing_fields = [
                    field
                    for field, value in (
                        ("docstring", query_text),
                        ("summary", summary_text),
                    )
                    if not isinstance(value, str) or not value.strip()
                ]
                if missing_fields:
                    logger.warning(
                        "Skipping training record %d in %s with empty or "
                        "missing fields: %s",
                        index,
                        file_path,
                        ", ".join(missing_fields),
                    )
                    continue
            elif text_field == "docstring" and (
                not isinstance(query_text, str) or not query_text.strip()
            ):
                raise ValueError(
                    "Record {} in {} is missing 'docstring'".format(
                        index,
                        file_path,
                    )
                )
            elif text_field == "summary" and (
                not isinstance(summary_text, str) or not summary_text.strip()
            ):
                logger.warning(
                    "Skipping record %d in %s without a summary",
                    index,
                    file_path,
                )
                continue

            query_tokens, query_ids = (
                encode_text(query_text, tokenizer, args.nl_length)
                if query_text
                else ([], [])
            )
            summary_tokens, summary_ids = (
                encode_text(summary_text, tokenizer, args.nl_length)
                if summary_text
                else ([], [])
            )
            self.examples.append(
                SummaryFeatures(
                    query_tokens,
                    query_ids,
                    summary_tokens,
                    summary_ids,
                    _record_url(record),
                )
            )

        if not self.examples:
            raise ValueError(
                "No usable query-summary examples found in {}".format(file_path)
            )

        if "train" in file_path:
            for index, example in enumerate(self.examples[:3]):
                logger.info("*** Query-summary example %d ***", index)
                logger.info("query_tokens: %s", example.query_tokens)
                logger.info("summary_tokens: %s", example.summary_tokens)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        return (
            torch.tensor(example.query_ids),
            torch.tensor(example.summary_ids),
        )


def train(args, model, model2, tokenizer):
    train_dataset = SummaryDataset(
        tokenizer,
        args,
        args.train_data_file,
        "pair",
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
    )

    optimizer = AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.learning_rate,
        eps=1e-8,
    )
    total_steps = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    logger.info("***** Running summary encoder training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total optimization steps = %d", total_steps)

    model.zero_grad()
    model2.zero_grad()
    model.train()
    model2.eval()

    total_loss = 0.0
    logged_steps = 0
    best_mrr = float("-inf")

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            query_inputs = batch[0].to(args.device)
            summary_inputs = batch[1].to(args.device)

            query_vec = model(query_inputs)
            summary_vec = model(summary_inputs)
            momentum_query_vec = model2(query_inputs)
            momentum_summary_vec = model2(summary_inputs)

            loss = (
                loss_fn(query_vec, summary_vec, args)
                + loss_fn(query_vec, momentum_summary_vec, args)
                + loss_fn(momentum_query_vec, summary_vec, args)
            )
            if args.covariance_loss:
                loss += covariance_loss(query_vec)

            total_loss += loss.item()
            logged_steps += 1
            if (step + 1) % 100 == 0:
                logger.info(
                    "epoch %d step %d loss %.5f",
                    epoch,
                    step + 1,
                    total_loss / logged_steps,
                )
                total_loss = 0.0
                logged_steps = 0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            with torch.no_grad():
                online_model = model.module if hasattr(model, "module") else model
                momentum_model = (
                    model2.module if hasattr(model2, "module") else model2
                )
                for online_parameter, momentum_parameter in zip(
                    online_model.parameters(),
                    momentum_model.parameters(),
                ):
                    momentum_parameter.data.mul_(args.moco_m).add_(
                        online_parameter.data,
                        alpha=1.0 - args.moco_m,
                    )

        results = evaluate(
            args,
            model,
            tokenizer,
            args.eval_data_file,
        )
        for key, value in results.items():
            logger.info("  %s = %.4f", key, value)

        if results["eval_mrr"] > best_mrr:
            best_mrr = results["eval_mrr"]
            logger.info("Best summary MRR: %.4f", best_mrr)
            save_best_model(args, model, tokenizer)


def best_checkpoint_dir(args):
    return os.path.join(args.output_dir, "checkpoint-best-mrr")


def load_best_encoder(args, model):
    checkpoint_dir = best_checkpoint_dir(args)
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.encoder = RobertaModel.from_pretrained(checkpoint_dir)
    model.to(args.device)
    logger.info("Loaded best summary encoder from %s", checkpoint_dir)


def _decode(tokens, tokenizer):
    excluded = {
        tokenizer.cls_token,
        tokenizer.sep_token,
        "<encoder-only>",
        tokenizer.pad_token,
    }
    return tokenizer.convert_tokens_to_string(
        [token for token in tokens if token not in excluded]
    ).strip()


def evaluate(args, model, tokenizer, file_name, export_path=None):
    query_dataset = SummaryDataset(
        tokenizer,
        args,
        file_name,
        "docstring",
    )
    summary_dataset = SummaryDataset(
        tokenizer,
        args,
        args.codebase_file,
        "summary",
    )
    query_dataloader = DataLoader(
        query_dataset,
        sampler=SequentialSampler(query_dataset),
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    summary_dataloader = DataLoader(
        summary_dataset,
        sampler=SequentialSampler(summary_dataset),
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )

    logger.info("***** Running summary evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num summaries = %d", len(summary_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    query_vecs = []
    summary_vecs = []
    with torch.no_grad():
        for batch in query_dataloader:
            query_vecs.append(
                model(batch[0].to(args.device)).cpu().numpy()
            )
        for batch in summary_dataloader:
            summary_vecs.append(
                model(batch[1].to(args.device)).cpu().numpy()
            )
    model.train()

    query_vecs = np.concatenate(query_vecs, axis=0)
    summary_vecs = np.concatenate(summary_vecs, axis=0)
    scores = np.matmul(query_vecs, summary_vecs.T)
    sort_ids = np.argsort(scores, axis=-1, kind="quicksort")[:, ::-1]

    query_urls = [example.url for example in query_dataset.examples]
    summary_urls = [example.url for example in summary_dataset.examples]
    reciprocal_ranks = []
    for url, candidate_ids in zip(query_urls, sort_ids):
        rank = next(
            (
                rank
                for rank, candidate_id in enumerate(candidate_ids[:1000], 1)
                if summary_urls[candidate_id] == url
            ),
            None,
        )
        reciprocal_ranks.append(1.0 / rank if rank is not None else 0.0)

    if export_path:
        rows = []
        topk = min(args.topk, len(summary_dataset))
        for query_index, candidate_ids in enumerate(sort_ids[:, :topk]):
            query_example = query_dataset.examples[query_index]
            for rank, summary_index in enumerate(candidate_ids, start=1):
                summary_index = int(summary_index)
                summary_example = summary_dataset.examples[summary_index]
                rows.append(
                    {
                        "query_index": query_index,
                        "query_url": query_urls[query_index],
                        "query": _decode(query_example.query_tokens, tokenizer),
                        "summary_index": summary_index,
                        "summary_url": summary_urls[summary_index],
                        "summary": _decode(
                            summary_example.summary_tokens,
                            tokenizer,
                        ),
                        "dual_encoder_topk_rank": rank,
                        "query_summary_score": float(
                            scores[query_index, summary_index]
                        ),
                        "is_positive": int(
                            summary_urls[summary_index] == query_urls[query_index]
                        ),
                    }
                )

        output_directory = os.path.dirname(os.path.abspath(export_path))
        os.makedirs(output_directory, exist_ok=True)
        fields = [
            "query_index",
            "query_url",
            "query",
            "summary_index",
            "summary_url",
            "summary",
            "dual_encoder_topk_rank",
            "query_summary_score",
            "is_positive",
        ]
        with open(
            export_path,
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Exported %d query-summary pairs to %s", len(rows), export_path)

    return {"eval_mrr": float(np.mean(reciprocal_ranks))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_file", default=None, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--eval_data_file", default=None, type=str)
    parser.add_argument("--test_data_file", default=None, type=str)
    parser.add_argument("--codebase_file", default=None, type=str)
    parser.add_argument("--test_results_file", default=None, type=str)
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--frozen_layers", default=None, type=str)
    parser.add_argument("--moco_m", default=0.999, type=float)
    parser.add_argument("--nl_length", default=128, type=int)
    parser.add_argument("--topk", default=100, type=int)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--do_zero_shot", action="store_true")
    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--eval_batch_size", default=4, type=int)
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of data loader worker processes.",
    )
    parser.add_argument(
        "--loss_type",
        default="poly",
        choices=["poly", "infonce", "adaptive_hard"],
        type=str,
    )
    parser.add_argument("--temperature", default=0.05, type=float)
    parser.add_argument("--margin_scale", default=0.2, type=float)
    parser.add_argument("--hard_k", default=2, type=int)
    parser.add_argument("--covariance_loss", action="store_true")
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    if args.do_test and args.topk <= 0:
        parser.error("--topk must be greater than 0")
    if args.num_workers < 0:
        parser.error("--num_workers must not be negative")

    print(json.dumps(vars(args), indent=4))
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    logger.info("device: %s, n_gpu: %s", args.device, args.n_gpu)
    set_seed(args.seed)
    args.frozen_layers = (
        args.frozen_layers.split(",") if args.frozen_layers else []
    )

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    model = Model(
        RobertaModel.from_pretrained(args.model_name_or_path),
        args,
    )
    model2 = Model(
        RobertaModel.from_pretrained(args.model_name_or_path),
        args,
    )
    for parameter in model2.parameters():
        parameter.requires_grad = False

    model.to(args.device)
    model2.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        model2 = torch.nn.DataParallel(model2)

    if args.do_train:
        train(args, model, model2, tokenizer)

    if args.do_eval:
        if not args.do_zero_shot:
            load_best_encoder(args, model)
        result = evaluate(args, model, tokenizer, args.eval_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result):
            logger.info("  %s = %.3f", key, result[key])

    if args.do_test:
        if not args.do_zero_shot:
            load_best_encoder(args, model)
        result = evaluate(
            args,
            model,
            tokenizer,
            args.test_data_file,
            export_path=args.test_results_file or os.path.join(
                args.output_dir,
                "dual_encoder_summary_test_results.csv",
            ),
        )
        logger.info("***** Test results *****")
        for key in sorted(result):
            logger.info("  %s = %.3f", key, result[key])


if __name__ == "__main__":
    main()
