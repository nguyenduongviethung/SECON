# coding=utf-8
"""Evaluate code and summary encoders with weighted score fusion."""

import argparse
import csv
import json
import logging
import os
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import RobertaModel, RobertaTokenizer

from run_encoder import Model, encode_text


logger = logging.getLogger(__name__)


def read_records(file_path):
    with open(file_path, encoding="utf-8") as data_file:
        if file_path.lower().endswith(".jsonl"):
            return [
                json.loads(line)
                for line in data_file
                if line.strip()
            ]
        if file_path.lower().endswith(".json"):
            return json.load(data_file)
    raise ValueError("Dataset must be a JSON or JSONL file: {}".format(file_path))


def record_url(record, index, file_path):
    url = record.get("url", record.get("retrieval_idx"))
    if url is None:
        raise ValueError(
            "Record {} in {} is missing 'url' and 'retrieval_idx'".format(
                index,
                file_path,
            )
        )
    return url


class RetrievalDataset(Dataset):
    """Pre-tokenized queries and code candidates for both encoders."""

    def __init__(
        self,
        file_path,
        code_tokenizer,
        summary_tokenizer,
        args,
        include_code,
        include_summary,
    ):
        self.examples = []
        records = read_records(file_path)
        for index, record in enumerate(records):
            docstring = record.get("docstring")
            if not isinstance(docstring, str) or not docstring.strip():
                if include_code:
                    docstring = ""
                else:
                    raise ValueError(
                        "Query record {} in {} is missing a non-empty "
                        "'docstring'".format(index, file_path)
                    )

            if docstring:
                _, query_code_ids = encode_text(
                    docstring,
                    code_tokenizer,
                    args.nl_length,
                )
                _, query_summary_ids = encode_text(
                    docstring,
                    summary_tokenizer,
                    args.nl_length,
                )
            else:
                query_code_ids = (
                    [code_tokenizer.pad_token_id] * args.nl_length
                )
                query_summary_ids = (
                    [summary_tokenizer.pad_token_id] * args.nl_length
                )

            code_ids = [code_tokenizer.pad_token_id] * args.code_length
            if include_code:
                code = record.get("code")
                if not isinstance(code, str) or not code.strip():
                    raise ValueError(
                        "Record {} in {} is missing a non-empty 'code'".format(
                            index,
                            file_path,
                        )
                    )
                _, code_ids = encode_text(
                    code,
                    code_tokenizer,
                    args.code_length,
                )

            summary = record.get("summary")
            has_summary = (
                isinstance(summary, str)
                and bool(summary.strip())
                and include_summary
            )
            if has_summary:
                _, summary_ids = encode_text(
                    summary,
                    summary_tokenizer,
                    args.nl_length,
                )
            else:
                summary_ids = [summary_tokenizer.pad_token_id] * args.nl_length

            self.examples.append(
                {
                    "url": record_url(record, index, file_path),
                    "docstring": docstring,
                    "code": record.get("code", ""),
                    "summary": summary if isinstance(summary, str) else "",
                    "code_ids": code_ids,
                    "query_code_ids": query_code_ids,
                    "query_summary_ids": query_summary_ids,
                    "summary_ids": summary_ids,
                    "has_summary": has_summary,
                }
            )

        if not self.examples:
            raise ValueError("No records found in {}".format(file_path))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        return (
            torch.tensor(example["code_ids"]),
            torch.tensor(example["query_code_ids"]),
            torch.tensor(example["query_summary_ids"]),
            torch.tensor(example["summary_ids"]),
            torch.tensor(example["has_summary"]),
        )


def encode(model, dataloader, args, input_index):
    vectors = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[input_index].to(args.device)
            vectors.append(model(inputs).cpu().numpy())
    return np.concatenate(vectors, axis=0)


def evaluate(args, code_model, summary_model, code_tokenizer, summary_tokenizer):
    query_dataset = RetrievalDataset(
        args.eval_data_file,
        code_tokenizer,
        summary_tokenizer,
        args,
        include_code=False,
        include_summary=False,
    )
    codebase_dataset = RetrievalDataset(
        args.codebase_file,
        code_tokenizer,
        summary_tokenizer,
        args,
        include_code=True,
        include_summary=True,
    )
    query_loader = DataLoader(
        query_dataset,
        sampler=SequentialSampler(query_dataset),
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    codebase_loader = DataLoader(
        codebase_dataset,
        sampler=SequentialSampler(codebase_dataset),
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )

    logger.info("***** Running weighted encoder evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num code candidates = %d", len(codebase_dataset))
    logger.info(
        "  Score weights: query-code=%.4f, query-summary=%.4f",
        args.query_code_weight,
        args.query_summary_weight,
    )

    code_model.eval()
    summary_model.eval()

    code_vecs = encode(code_model, codebase_loader, args, input_index=0)
    query_code_vecs = encode(code_model, query_loader, args, input_index=1)
    query_summary_vecs = encode(
        summary_model,
        query_loader,
        args,
        input_index=2,
    )
    summary_vecs = encode(summary_model, codebase_loader, args, input_index=3)

    code_scores = np.matmul(query_code_vecs, code_vecs.T)
    summary_scores = np.matmul(query_summary_vecs, summary_vecs.T)
    summary_mask = np.asarray(
        [example["has_summary"] for example in codebase_dataset.examples],
        dtype=bool,
    )

    total_weight = args.query_code_weight + args.query_summary_weight
    scores = (
        args.query_code_weight * code_scores
        + args.query_summary_weight * summary_scores
    ) / total_weight
    scores[:, ~summary_mask] = code_scores[:, ~summary_mask]

    sorted_indices = np.argsort(
        scores,
        axis=-1,
        kind="quicksort",
    )[:, ::-1]
    query_urls = [example["url"] for example in query_dataset.examples]
    code_urls = [example["url"] for example in codebase_dataset.examples]

    reciprocal_ranks = []
    for query_url, candidate_indices in zip(query_urls, sorted_indices):
        rank = next(
            (
                position
                for position, candidate_index in enumerate(
                    candidate_indices[:1000],
                    start=1,
                )
                if code_urls[candidate_index] == query_url
            ),
            None,
        )
        reciprocal_ranks.append(1.0 / rank if rank is not None else 0.0)
    mrr = float(np.mean(reciprocal_ranks))

    if args.output_file:
        rows = []
        topk = min(args.topk, len(codebase_dataset))
        for query_index, candidate_indices in enumerate(
            sorted_indices[:, :topk]
        ):
            query_text = query_dataset.examples[query_index]["docstring"]
            query_url = query_urls[query_index]
            for rank, candidate_index in enumerate(candidate_indices, start=1):
                candidate_index = int(candidate_index)
                candidate = codebase_dataset.examples[candidate_index]
                rows.append(
                    {
                        "query_index": query_index,
                        "query_url": query_url,
                        "query": query_text,
                        "codebase_index": candidate_index,
                        "code_url": code_urls[candidate_index],
                        "code": candidate["code"],
                        "summary": candidate["summary"],
                        "dual_encoder_topk_rank": rank,
                        "query_code_score": float(
                            code_scores[query_index, candidate_index]
                        ),
                        "query_summary_score": (
                            float(summary_scores[query_index, candidate_index])
                            if summary_mask[candidate_index]
                            else ""
                        ),
                        "total_score": float(scores[query_index, candidate_index]),
                        "is_positive": int(code_urls[candidate_index] == query_url),
                    }
                )

        output_directory = os.path.dirname(os.path.abspath(args.output_file))
        os.makedirs(output_directory, exist_ok=True)
        fields = [
            "query_index",
            "query_url",
            "query",
            "codebase_index",
            "code_url",
            "code",
            "summary",
            "dual_encoder_topk_rank",
            "query_code_score",
            "query_summary_score",
            "total_score",
            "is_positive",
        ]
        with open(
            args.output_file,
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Exported %d ranked pairs to %s", len(rows), args.output_file)

    return {"eval_mrr": mrr}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate weighted query-code and query-summary scores."
    )
    parser.add_argument("--code_model_path", required=True, type=str)
    parser.add_argument(
        "--summary_model_path",
        default=None,
        type=str,
        help="Defaults to --code_model_path when omitted.",
    )
    parser.add_argument("--eval_data_file", required=True, type=str)
    parser.add_argument("--codebase_file", required=True, type=str)
    parser.add_argument("--output_file", default=None, type=str)
    parser.add_argument("--nl_length", default=128, type=int)
    parser.add_argument("--code_length", default=256, type=int)
    parser.add_argument("--eval_batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--topk", default=100, type=int)
    parser.add_argument("--query_code_weight", default=0.5, type=float)
    parser.add_argument("--query_summary_weight", default=0.5, type=float)
    args = parser.parse_args()
    if args.summary_model_path is None:
        args.summary_model_path = args.code_model_path

    weights = (args.query_code_weight, args.query_summary_weight)
    if (
        not all(np.isfinite(weight) and weight >= 0 for weight in weights)
        or sum(weights) <= 0
    ):
        parser.error(
            "Score weights must be finite and non-negative, "
            "with at least one greater than zero."
        )
    if args.eval_batch_size <= 0:
        parser.error("--eval_batch_size must be greater than 0")
    if args.num_workers < 0:
        parser.error("--num_workers must not be negative")
    if args.topk <= 0:
        parser.error("--topk must be greater than 0")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", args.device)

    code_tokenizer = RobertaTokenizer.from_pretrained(args.code_model_path)
    summary_tokenizer = RobertaTokenizer.from_pretrained(args.summary_model_path)
    model_args = SimpleNamespace(frozen_layers=[])
    code_model = Model(
        RobertaModel.from_pretrained(args.code_model_path),
        model_args,
    ).to(args.device)
    summary_model = Model(
        RobertaModel.from_pretrained(args.summary_model_path),
        model_args,
    ).to(args.device)

    result = evaluate(
        args,
        code_model,
        summary_model,
        code_tokenizer,
        summary_tokenizer,
    )
    logger.info("***** Evaluation results *****")
    logger.info("  eval_mrr = %.4f", result["eval_mrr"])


if __name__ == "__main__":
    main()
