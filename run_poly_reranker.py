#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual Encoder -> Top-K -> Poly Attention -> MLP Reranker

Pipeline
--------
1. Load ONE dual-encoder checkpoint/model_name_or_path.
2. Load the same backbone into the original Model for sequence embeddings.
3. Use the frozen dual encoder to retrieve top-k candidates.
4. Load the same backbone into TokenEmbeddingModel and extract token-level
   contextual embeddings once. These embeddings are kept in CPU RAM.
5. For every query/top-k-code pair, train only:
       - PolyEncoder.poly_code_embeddings
       - MLP reranker
   The Transformer encoder itself remains frozen.
6. Select the best checkpoint by validation MRR.
7. Evaluate/test with the cached token embeddings and the trained
   PolyAttention + MLP reranker.

Poly attention
--------------
For each candidate code:
    code tokens [L, H]
        -> M learned poly codes attend over code tokens
        -> M global code vectors [M, H]

For each query/candidate pair:
    query CLS [H]
        -> attends over the M global code vectors
        -> pair representation [H]
        -> MLP -> reranking score

The cached token embeddings are detached CPU tensors. Therefore no gradient
is propagated through the dual/token encoder. Only poly codes and MLP are
optimized.
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
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from transformers import RobertaModel, RobertaTokenizer

from run_no_poly import Model, TextDataset

logger = logging.getLogger(__name__)


# ============================================================
# Utilities
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
    torch.backends.cudnn.benchmark = False


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def load_state_dict(model, checkpoint, device):
    logger.info("Loading checkpoint: %s", checkpoint)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    logger.info("Checkpoint loaded.")


def best_reranker_checkpoint(args):
    return os.path.join(
        args.output_dir,
        "checkpoint-best-mrr",
        "poly_mlp.bin",
    )


def save_best_reranker(args, reranker):
    checkpoint = best_reranker_checkpoint(args)
    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
    torch.save(unwrap_model(reranker).state_dict(), checkpoint)
    logger.info("Saved best Poly+MLP checkpoint to %s", checkpoint)


def load_reranker_checkpoint(reranker, checkpoint, device):
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError("Reranker checkpoint not found: %s" % checkpoint)
    load_state_dict(unwrap_model(reranker), checkpoint, device)


# ============================================================
# Loss
# ============================================================

def adaptive_hard_rerank_loss(
    scores,
    labels,
    temperature=0.05,
    margin_scale=0.2,
    hard_k=2,
):
    """Adaptive hard-negative ranking loss used by run_reranker.py."""
    _, k_total = scores.shape
    positive_mask = labels.bool()
    valid = positive_mask.any(dim=1)

    if not valid.any():
        return scores.sum() * 0.0

    scores = scores[valid]
    positive_mask = positive_mask[valid]

    pos = scores.masked_select(positive_mask)
    neg = scores.masked_fill(positive_mask, -1e9)

    k = min(hard_k, k_total - 1)
    hard_neg, _ = torch.topk(neg, k=k, dim=1)

    margin = margin_scale * (1.0 - torch.sigmoid(pos))
    weights = F.softmax(hard_neg / temperature, dim=1)

    ranking_loss = F.softplus(
        (hard_neg - pos.unsqueeze(1) + margin.unsqueeze(1)) / temperature
    )

    return (weights * ranking_loss).sum(dim=1).mean()


# ============================================================
# Dataset
# ============================================================

def build_dataset(tokenizer, args, query_file, code_file=None, is_train=False):
    query_dataset = TextDataset(tokenizer, args, query_file)
    if is_train:
        code_dataset = TextDataset(tokenizer, args, query_file)
    else:
        code_dataset = TextDataset(tokenizer, args, code_file)
    return query_dataset, code_dataset


def dataset_tensors(dataset):
    code_ids = torch.tensor([x.code_ids for x in dataset.examples], dtype=torch.long)
    nl_ids = torch.tensor([x.nl_ids for x in dataset.examples], dtype=torch.long)
    urls = [x.url for x in dataset.examples]
    return code_ids, nl_ids, urls


# ============================================================
# Frozen dual encoder: sequence embeddings for retrieval
# ============================================================

@torch.no_grad()
def encode_dataset(model, dataset, args, is_query):
    sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.dual_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model.eval()
    embeddings = []

    for batch in dataloader:
        inputs = batch[1] if is_query else batch[0]
        inputs = inputs.to(args.device, non_blocking=True)
        vec = model(nl_inputs=inputs) if is_query else model(code_inputs=inputs)
        embeddings.append(vec.detach().cpu())

    return torch.cat(embeddings, dim=0)


@torch.no_grad()
def retrieve_topk(query_embeddings, code_embeddings, topk, args):
    query_embeddings = F.normalize(query_embeddings, dim=-1)
    code_embeddings = F.normalize(code_embeddings, dim=-1)

    all_indices = []
    all_scores = []
    code_gpu = code_embeddings.to(args.device)

    for start in range(0, query_embeddings.size(0), args.retrieval_batch_size):
        end = min(start + args.retrieval_batch_size, query_embeddings.size(0))
        q = query_embeddings[start:end].to(args.device)
        scores = torch.matmul(q, code_gpu.transpose(0, 1))
        k = min(topk, scores.size(1))
        values, indices = torch.topk(scores, k=k, dim=1)
        all_scores.append(values.cpu())
        all_indices.append(indices.cpu())

    return torch.cat(all_indices, 0), torch.cat(all_scores, 0)


# ============================================================
# Token embedding cache
# ============================================================

class TokenEmbeddingModel(nn.Module):
    """Frozen dual-encoder backbone returning contextual token embeddings."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, inputs):
        mask = inputs.ne(1)
        hidden = self.encoder(inputs, attention_mask=mask)[0]
        return hidden, mask


@torch.no_grad()
def build_token_embedding_cache(model, dataset, args, is_query, dtype):
    """Encode one complete dataset once and keep [N,L,H] in CPU RAM."""
    code_ids, nl_ids, _ = dataset_tensors(dataset)
    ids = nl_ids if is_query else code_ids

    dataloader = DataLoader(
        TensorDataset(ids),
        batch_size=args.token_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model.eval()
    embeddings = []
    masks = []

    for step, (batch_ids,) in enumerate(dataloader):
        batch_ids = batch_ids.to(args.device, non_blocking=True)
        hidden, mask = model(batch_ids)
        embeddings.append(hidden.detach().to(dtype=dtype).cpu())
        masks.append(mask.detach().cpu())

        if (step + 1) % args.log_steps == 0:
            cached_rows = sum(x.size(0) for x in embeddings)
            logger.info(
                "Token encoding %d batches; cached rows = %d",
                step + 1,
                cached_rows,
            )

    return torch.cat(embeddings, 0), torch.cat(masks, 0)


# ============================================================
# Poly attention + MLP
# ============================================================

class PolyAttention(nn.Module):
    """Poly-encoder style code-side global representation + query attention."""

    def __init__(self, hidden_size, poly_m=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.poly_m = poly_m
        self.poly_code_embeddings = nn.Embedding(poly_m, hidden_size)
        nn.init.normal_(
            self.poly_code_embeddings.weight,
            mean=0.0,
            std=hidden_size ** -0.5,
        )

    def code_global(self, code_tokens, code_mask):
        """
        code_tokens: [B,K,L,H]
        code_mask:   [B,K,L]
        returns:     [B,K,M,H]
        """
        bsz, topk, length, hidden = code_tokens.shape
        poly = self.poly_code_embeddings.weight
        poly = poly.view(1, 1, self.poly_m, hidden).expand(bsz, topk, -1, -1)

        logits = torch.einsum("bkmh,bklh->bkml", poly, code_tokens)
        logits = logits / (hidden ** 0.5)
        logits = logits.masked_fill(~code_mask.unsqueeze(2), -1e4)
        weights = F.softmax(logits, dim=-1)
        return torch.einsum("bkml,bklh->bkmh", weights, code_tokens)

    def query_attend(self, query_mean, poly_repr):
        """
        query_mean: [B,H]
        poly_repr:  [B,K,M,H]
        returns:    [B,K,H]
        """
        logits = torch.einsum("bh,bkmh->bkm", query_mean, poly_repr)
        logits = logits / (self.hidden_size ** 0.5)
        weights = F.softmax(logits, dim=-1)
        return torch.einsum("bkm,bkmh->bkh", weights, poly_repr)

    def forward(self, query_tokens, query_mask, code_tokens, code_mask):
        # CPU cache may be float16 to reduce RAM. Poly parameters remain in
        # the module dtype (normally float32), so cast cached embeddings only
        # after they are moved to the attention device.
        dtype = self.poly_code_embeddings.weight.dtype
        query_tokens = query_tokens.to(dtype=dtype)
        code_tokens = code_tokens.to(dtype=dtype)
        query_mask = query_mask.to(dtype=dtype)
        query_mean = (
            (query_tokens * query_mask.unsqueeze(-1)).sum(dim=1)
            / query_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        )
        poly_repr = self.code_global(code_tokens, code_mask)
        return self.query_attend(query_mean, poly_repr)


class MLPReranker(nn.Module):
    def __init__(self, input_dim, hidden_dims=(512,), dropout=0.1):
        super().__init__()
        layers = []
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PolyMLPReranker(nn.Module):
    """Trainable part: Poly code embeddings + MLP."""

    def __init__(self, hidden_size, poly_m, mlp_hidden_dims, mlp_dropout):
        super().__init__()
        self.poly = PolyAttention(hidden_size, poly_m)
        self.mlp = MLPReranker(hidden_size, mlp_hidden_dims, mlp_dropout)

    def forward(self, query_tokens, query_mask, code_tokens, code_mask):
        pair_repr = self.poly(
            query_tokens,
            query_mask,
            code_tokens,
            code_mask,
        )
        bsz, topk, hidden = pair_repr.shape
        scores = self.mlp(pair_repr.reshape(bsz * topk, hidden))
        return scores.view(bsz, topk)


# ============================================================
# Cache construction: dual retrieval + token embeddings
# ============================================================

@torch.no_grad()
def prepare_poly_cache(
    args,
    tokenizer,
    dual_model,
    token_model,
    query_file,
    code_file,
    is_train=False,
    token_dtype=torch.float32,
):
    query_dataset, code_dataset = build_dataset(
        tokenizer, args, query_file, code_file, is_train=is_train
    )

    logger.info("Queries: %d", len(query_dataset))
    logger.info("Codes: %d", len(code_dataset))

    # 1. Original Model -> sequence embeddings -> top-k.
    query_embeddings = encode_dataset(dual_model, query_dataset, args, is_query=True)
    code_embeddings = encode_dataset(dual_model, code_dataset, args, is_query=False)
    topk_indices, topk_scores = retrieve_topk(
        query_embeddings, code_embeddings, args.topk, args
    )

    # 2. Same dual checkpoint loaded into TokenEmbeddingModel -> token cache.
    query_tokens, query_mask = build_token_embedding_cache(
        token_model, query_dataset, args, is_query=True, dtype=token_dtype
    )
    code_tokens, code_mask = build_token_embedding_cache(
        token_model, code_dataset, args, is_query=False, dtype=token_dtype
    )

    logger.info("Query token cache: %s (%s)", tuple(query_tokens.shape), query_tokens.dtype)
    logger.info("Code token cache:  %s (%s)", tuple(code_tokens.shape), code_tokens.dtype)
    logger.info("Top-k indices:      %s", tuple(topk_indices.shape))

    return {
        "query_dataset": query_dataset,
        "code_dataset": code_dataset,
        "topk_indices": topk_indices,
        "topk_scores": topk_scores,
        "query_tokens": query_tokens,
        "query_mask": query_mask,
        "code_tokens": code_tokens,
        "code_mask": code_mask,
    }


# ============================================================
# Labels
# ============================================================

def build_labels(cache):
    query_urls = [x.url for x in cache["query_dataset"].examples]
    code_urls = [x.url for x in cache["code_dataset"].examples]
    topk_indices = cache["topk_indices"]

    labels = torch.zeros(topk_indices.size(), dtype=torch.float32)
    for i, query_url in enumerate(query_urls):
        for j in range(topk_indices.size(1)):
            code_idx = int(topk_indices[i, j])
            labels[i, j] = float(code_urls[code_idx] == query_url)
    return labels


# ============================================================
# Batch materialization from CPU RAM
# ============================================================

def materialize_poly_batch(cache, query_indices, device):
    """
    Gather query token embeddings and the top-k code token embeddings from CPU
    RAM only when the MLP/Poly batch is processed.
    """
    topk_indices = cache["topk_indices"][query_indices]

    query_tokens = cache["query_tokens"][query_indices]
    query_mask = cache["query_mask"][query_indices]

    # [B,K,L,H]
    code_tokens = cache["code_tokens"][topk_indices]
    code_mask = cache["code_mask"][topk_indices]

    return (
        query_tokens.to(device, non_blocking=True),
        query_mask.to(device, non_blocking=True),
        code_tokens.to(device, non_blocking=True),
        code_mask.to(device, non_blocking=True),
    )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_cached(args, reranker, cache):
    reranker.eval()
    scores_all = []
    n = cache["topk_indices"].size(0)

    for start in range(0, n, args.mlp_inference_batch_size):
        end = min(start + args.mlp_inference_batch_size, n)
        indices = torch.arange(start, end)
        qtok, qmask, ctok, cmask = materialize_poly_batch(
            cache,
            indices,
            args.device,
        )
        scores_all.append(reranker(qtok, qmask, ctok, cmask).cpu())

    scores_all = torch.cat(scores_all, dim=0)
    query_urls = [x.url for x in cache["query_dataset"].examples]
    code_urls = [x.url for x in cache["code_dataset"].examples]
    topk_indices = cache["topk_indices"]

    ranks = []
    hits = []

    for i, query_url in enumerate(query_urls):
        order = torch.argsort(scores_all[i], descending=True)
        rank = None
        for position, candidate_position in enumerate(order.tolist(), start=1):
            code_idx = int(topk_indices[i, candidate_position])
            if code_urls[code_idx] == query_url:
                rank = position
                break

        ranks.append(0.0 if rank is None else 1.0 / rank)
        hits.append(float(any(
            code_urls[int(x)] == query_url for x in topk_indices[i]
        )))

    return {
        "mrr": float(np.mean(ranks)),
        "topk_recall": float(np.mean(hits)),
    }


# ============================================================
# Training
# ============================================================

def train_poly_mlp(args, reranker, train_cache, labels, validation_cache):
    # We sample query indices, not individual pairs, so every loss still sees
    # the complete [B,K] candidate set.
    dataset = TensorDataset(torch.arange(labels.size(0)), labels)
    dataloader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset),
        batch_size=args.mlp_batch_size,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = AdamW(
        [p for p in reranker.parameters() if p.requires_grad],
        lr=args.learning_rate,
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    logger.info("***** Training Poly Attention + MLP *****")
    logger.info("Queries = %d", labels.size(0))
    logger.info("Top-K = %d", labels.size(1))
    logger.info("Poly codes = %d", args.poly_m)
    base_reranker = reranker.module if isinstance(reranker, nn.DataParallel) else reranker
    logger.info("Hidden size = %d", base_reranker.poly.hidden_size)

    best_mrr = float("-inf")

    for epoch in range(args.num_train_epochs):
        reranker.train()
        total_loss = 0.0
        total_steps = 0

        for step, (query_indices, y) in enumerate(dataloader):
            query_indices = query_indices.long()
            y = y.to(args.device, non_blocking=True)

            qtok, qmask, ctok, cmask = materialize_poly_batch(
                train_cache, query_indices, args.device
            )

            scores = reranker(qtok, qmask, ctok, cmask)
            loss = adaptive_hard_rerank_loss(
                scores,
                y,
                temperature=args.temperature,
                margin_scale=args.margin_scale,
                hard_k=args.hard_k,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                reranker.parameters(), args.max_grad_norm
            )
            optimizer.step()

            total_loss += loss.item()
            total_steps += 1

            if (step + 1) % args.log_steps == 0:
                logger.info(
                    "epoch %d step %d loss %.6f",
                    epoch,
                    step + 1,
                    total_loss / total_steps,
                )

        epoch_loss = total_loss / max(total_steps, 1)
        result = evaluate_cached(args, reranker, validation_cache)
        logger.info(
            "Epoch %d loss %.6f | validation MRR %.6f | top-k recall %.6f",
            epoch,
            epoch_loss,
            result["mrr"],
            result["topk_recall"],
        )

        if result["mrr"] > best_mrr:
            best_mrr = result["mrr"]
            save_best_reranker(args, reranker)


# ============================================================
# Model construction
# ============================================================

def prepare_models(args, tokenizer):
    """Load the SAME dual checkpoint twice: pooled Model + token Model."""
    model_path = args.model_name_or_path
    if not args.do_zero_shot and args.dual_checkpoint:
        model_path = args.dual_checkpoint

    logger.info("Dual/token encoder path: %s", model_path)

    # First copy: original Model -> sequence embedding -> retrieval.
    dual_backbone = RobertaModel.from_pretrained(model_path)
    dual_model = Model(dual_backbone, args)

    # Second copy: TokenEmbeddingModel -> token embedding cache.
    token_backbone = RobertaModel.from_pretrained(model_path)
    token_model = TokenEmbeddingModel(token_backbone)

    for p in dual_model.parameters():
        p.requires_grad = False
    for p in token_model.parameters():
        p.requires_grad = False

    dual_model.eval()
    token_model.eval()

    hidden_size = dual_backbone.config.hidden_size
    reranker = PolyMLPReranker(
        hidden_size=hidden_size,
        poly_m=args.poly_m,
        mlp_hidden_dims=args.mlp_hidden_dims,
        mlp_dropout=args.mlp_dropout,
    )

    if args.reranker_checkpoint and os.path.isfile(args.reranker_checkpoint):
        load_reranker_checkpoint(
            reranker, args.reranker_checkpoint, args.device
        )

    dual_model.to(args.device)
    token_model.to(args.device)
    reranker.to(args.device)

    if args.n_gpu > 1:
        dual_model = nn.DataParallel(dual_model)
        token_model = nn.DataParallel(token_model)
        reranker = nn.DataParallel(reranker)

    return dual_model, token_model, reranker


# ============================================================
# Stage
# ============================================================

def run_stage(
    args,
    tokenizer,
    dual_model,
    token_model,
    reranker,
    query_file,
    code_file,
    is_train,
    validation_file=None,
    validation_code_file=None,
):
    dtype = torch.float16 if args.token_cache_dtype == "float16" else torch.float32

    cache = prepare_poly_cache(
        args,
        tokenizer,
        dual_model,
        token_model,
        query_file,
        code_file,
        is_train=is_train,
        token_dtype=dtype,
    )

    if is_train:
        validation_cache = prepare_poly_cache(
            args,
            tokenizer,
            dual_model,
            token_model,
            validation_file,
            validation_code_file,
            is_train=False,
            token_dtype=dtype,
        )
        labels = build_labels(cache)
        train_poly_mlp(args, reranker, cache, labels, validation_cache)
        return None

    return evaluate_cached(args, reranker, cache)


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--train_data_file", type=str, default=None)
    parser.add_argument("--eval_data_file", type=str, default=None)
    parser.add_argument("--test_data_file", type=str, default=None)
    parser.add_argument("--codebase_file", type=str, default=None)

    # One dual encoder only
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="HuggingFace model path for the single dual encoder.",
    )
    parser.add_argument(
        "--dual_checkpoint",
        type=str,
        default=None,
        help="Optional trained dual-encoder checkpoint/directory.",
    )
    parser.add_argument("--reranker_checkpoint", type=str, default=None)

    # Modes
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--do_zero_shot", action="store_true")

    # Sequence lengths
    parser.add_argument("--nl_length", type=int, default=128)
    parser.add_argument("--code_length", type=int, default=256)

    # Retrieval
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--dual_batch_size", type=int, default=32)
    parser.add_argument("--retrieval_batch_size", type=int, default=32)

    # Token cache
    parser.add_argument(
        "--token_batch_size",
        type=int,
        default=32,
        help="Batch size used only while creating the RAM token cache.",
    )
    parser.add_argument(
        "--token_cache_dtype",
        choices=["float32", "float16"],
        default="float32",
        help="CPU cache dtype. float16 halves RAM use but may slightly change attention numerics.",
    )

    # Poly attention
    parser.add_argument("--poly_m", type=int, default=8)

    # MLP
    parser.add_argument("--mlp_hidden_dims", nargs="+", type=int, default=[512])
    parser.add_argument("--mlp_dropout", type=float, default=0.1)
    parser.add_argument("--mlp_batch_size", type=int, default=64)
    parser.add_argument("--mlp_inference_batch_size", type=int, default=64)

    # Loss
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--margin_scale", type=float, default=0.2)
    parser.add_argument("--hard_k", type=int, default=2)

    # Training
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--log_steps", type=int, default=100)

    # Misc
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--frozen_layers", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if args.do_train:
        if args.train_data_file is None:
            raise ValueError("--train_data_file is required for --do_train")
        if args.eval_data_file is None:
            raise ValueError("--eval_data_file is required for validation")
        if args.codebase_file is None:
            raise ValueError("--codebase_file is required for validation")

    if args.do_eval and (args.eval_data_file is None or args.codebase_file is None):
        raise ValueError("--eval_data_file and --codebase_file are required for --do_eval")

    if args.do_test and (args.test_data_file is None or args.codebase_file is None):
        raise ValueError("--test_data_file and --codebase_file are required for --do_test")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.frozen_layers = args.frozen_layers.split(",") if args.frozen_layers else []

    set_seed(args.seed)

    logger.info("Arguments:\n%s", json.dumps(vars(args), indent=4, default=str))
    logger.info("device: %s, n_gpu: %d", args.device, args.n_gpu)

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    dual_model, token_model, reranker = prepare_models(args, tokenizer)

    if args.do_train:
        run_stage(
            args,
            tokenizer,
            dual_model,
            token_model,
            reranker,
            args.train_data_file,
            args.train_data_file,
            True,
            args.eval_data_file,
            args.codebase_file,
        )
        args.reranker_checkpoint = best_reranker_checkpoint(args)
        load_reranker_checkpoint(reranker, args.reranker_checkpoint, args.device)

    elif args.reranker_checkpoint is None and (args.do_eval or args.do_test):
        args.reranker_checkpoint = best_reranker_checkpoint(args)
        load_reranker_checkpoint(reranker, args.reranker_checkpoint, args.device)

    if args.do_eval:
        result = run_stage(
            args,
            tokenizer,
            dual_model,
            token_model,
            reranker,
            args.eval_data_file,
            args.codebase_file,
            False,
        )
        logger.info("***** Eval results *****")
        for key, value in result.items():
            logger.info("  %s = %.6f", key, value)

    if args.do_test:
        result = run_stage(
            args,
            tokenizer,
            dual_model,
            token_model,
            reranker,
            args.test_data_file,
            args.codebase_file,
            False,
        )
        logger.info("***** Test results *****")
        for key, value in result.items():
            logger.info("  %s = %.6f", key, value)


if __name__ == "__main__":
    main()
