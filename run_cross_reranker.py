# coding=utf-8
"""
Dual Encoder -> Top-K -> Cross Encoder -> MLP Reranker
Pipeline:
    1. Load a dual encoder:
         - from --dual_checkpoint, or
         - directly from HuggingFace with --do_zero_shot
    2. Encode all queries and all candidate codes with the FROZEN dual encoder.
    3. Retrieve top-k candidates using dual-encoder similarity.
    4. Train the cross encoder and MLP JOINTLY on query/top-k-code pairs.
       - Cross encoder and MLP both receive gradients.
       - Cross embeddings are recomputed every step (never cached) so the
         cross encoder can be updated.
    5. Select the best joint checkpoint by validation MRR.
    6. Evaluate/Test with the frozen dual encoder and trained cross+MLP.
Data:
    train:
        --train_data_file
        The same file contains both query/docstring and positive code.
        The complete train file is also used as the code candidate pool.
    eval/test:
        --eval_data_file / --test_data_file
        Query file.
        --codebase_file
        Candidate code pool.
Important:
    - Dual encoder is frozen.
    - Cross encoder AND MLP are jointly trainable.
    - Cross embeddings are recomputed during training (not cached), so
      gradients can update the cross encoder.
    - cross_batch_size is the number of queries processed at once
      inside the frozen cross-encoder feature builder (validation/eval).
      Each such batch therefore contains:
          cross_batch_size * topk
      query-code pairs.
    - mlp_batch_size is the number of queries processed per training step
      (before gradient accumulation). Each training step therefore
      contains mlp_batch_size * topk query-code pairs.
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


def best_mlp_checkpoint(args):
    return os.path.join(args.output_dir, "checkpoint-best-mrr", "mlp.bin")


def best_cross_checkpoint(args):
    return os.path.join(
        args.output_dir, "checkpoint-best-mrr", "cross_encoder.bin"
    )


def save_best_joint(args, cross_encoder, mlp):
    checkpoint_dir = os.path.join(args.output_dir, "checkpoint-best-mrr")
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(
        unwrap_model(cross_encoder).state_dict(),
        best_cross_checkpoint(args),
    )
    torch.save(
        unwrap_model(mlp).state_dict(),
        best_mlp_checkpoint(args),
    )
    logger.info(
        "Saved best validation-MRR cross encoder and MLP to %s", checkpoint_dir
    )


def save_best_mlp(args, mlp):
    checkpoint = best_mlp_checkpoint(args)
    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
    torch.save(unwrap_model(mlp).state_dict(), checkpoint)
    logger.info(
        "Saved best validation-MRR MLP checkpoint to %s", checkpoint
    )


def load_mlp_checkpoint(mlp, checkpoint, device):
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError("MLP checkpoint not found: %s" % checkpoint)
    load_state_dict(unwrap_model(mlp), checkpoint, device)


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
    """
    Adaptive hard-negative ranking loss for reranking.
    Args:
        scores:      [B, K] MLP scores.
        labels:      [B, K], 1 denotes the positive candidate.
        temperature: Controls hard-negative weighting / sharpness.
        margin_scale: Scale of the adaptive margin.
        hard_k:      Number of hardest negatives used per query.
    """
    B, K = scores.shape
    positive_mask = labels.bool()

    # Keep only queries that contain a positive candidate.
    valid = positive_mask.any(dim=1)
    if not valid.any():
        return scores.sum() * 0.0
    scores = scores[valid]
    positive_mask = positive_mask[valid]

    # Positive score [B_valid]
    pos = scores.masked_select(positive_mask)

    # Negative scores [B_valid, K]
    neg = scores.masked_fill(positive_mask, -1e9)

    # Select hardest negatives.
    num_neg = K - 1
    k = min(hard_k, num_neg)
    hard_neg, _ = torch.topk(neg, k=k, dim=1)

    # Adaptive margin: low positive confidence -> larger margin.
    margin = margin_scale * (1.0 - torch.sigmoid(pos))

    # Harder negatives receive larger weights.
    weights = F.softmax(hard_neg / temperature, dim=1)

    # Ranking objective: hard_neg < pos - margin.
    ranking_loss = F.softplus(
        (hard_neg - pos.unsqueeze(1) + margin.unsqueeze(1)) / temperature
    )

    loss = (weights * ranking_loss).sum(dim=1)
    return loss.mean()


# ============================================================
# Cross Encoder
# ============================================================
class CrossEncoder(nn.Module):
    """
    Cross encoder for query-code interaction.
    Input:
        query_ids: [B, Q]
        code_ids:  [B, C]
    The two sequences are merged into one RoBERTa sequence:
        [CLS] query [SEP] code [SEP]
    The pooled representation is the final hidden state of [CLS].
    """

    def __init__(self, encoder, tokenizer):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def build_pair_inputs(self, query_ids, code_ids):
        """
        Construct:
            [CLS] query [SEP] code [SEP]
        from already padded query/code tensors.
        Returns:
            pair_ids:       [B, L]
            attention_mask: [B, L]
        """
        device = query_ids.device
        batch_size = query_ids.size(0)
        sequences = []

        cls_t = torch.tensor(
            [self.cls_token_id], device=device, dtype=torch.long
        )
        sep_t = torch.tensor(
            [self.sep_token_id], device=device, dtype=torch.long
        )

        for i in range(batch_size):
            query = query_ids[i]
            code = code_ids[i]

            # Remove padding.
            query = query[query.ne(self.pad_token_id)]
            code = code[code.ne(self.pad_token_id)]

            # Remove existing special tokens if present.
            if query.numel() > 0:
                query = query[
                    (query != self.cls_token_id)
                    & (query != self.sep_token_id)
                ]
            if code.numel() > 0:
                code = code[
                    (code != self.cls_token_id)
                    & (code != self.sep_token_id)
                ]

            pair = torch.cat([cls_t, query, sep_t, code, sep_t])
            sequences.append(pair)

        max_length = max(x.size(0) for x in sequences)
        max_length = min(max_length, self.max_length)

        pair_ids = torch.full(
            (batch_size, max_length),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        for i, seq in enumerate(sequences):
            seq = seq[:max_length]
            pair_ids[i, : seq.size(0)] = seq

        attention_mask = pair_ids.ne(self.pad_token_id).long()
        return pair_ids, attention_mask

    def forward(self, query_ids, code_ids):
        pair_ids, attention_mask = self.build_pair_inputs(
            query_ids, code_ids
        )
        outputs = self.encoder(pair_ids, attention_mask=attention_mask)
        # [B, hidden_size]
        return outputs[0][:, 0, :]

    @property
    def max_length(self):
        return self._max_length

    @max_length.setter
    def max_length(self, value):
        self._max_length = value


# ============================================================
# MLP Reranker
# ============================================================
class MLPReranker(nn.Module):
    """
    Trainable reranking head.
    Input:  cross-encoder embedding [B, D]
    Output: logits [B]
    """

    def __init__(self, input_dim, hidden_dims=[512], dropout=0.1):
        super().__init__()
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================
# Dataset preparation
# ============================================================
def build_dataset(tokenizer, args, query_file, code_file=None, is_train=False):
    """
    Build query/code datasets.
    Train:
        query_file == train file
        code_file  == train file
    Eval/Test:
        query_file == eval/test file
        code_file  == codebase file
    """
    query_dataset = TextDataset(tokenizer, args, query_file)
    if is_train:
        code_dataset = TextDataset(tokenizer, args, query_file)
    else:
        code_dataset = TextDataset(tokenizer, args, code_file)
    return query_dataset, code_dataset


def dataset_tensors(dataset):
    """
    Extract tensors from TextDataset.
    Returns:
        code_ids: [N, code_length]
        nl_ids:   [N, nl_length]
        urls:     list[str]
    """
    code_ids = torch.tensor(
        [x.code_ids for x in dataset.examples], dtype=torch.long
    )
    nl_ids = torch.tensor(
        [x.nl_ids for x in dataset.examples], dtype=torch.long
    )
    urls = [x.url for x in dataset.examples]
    return code_ids, nl_ids, urls


# ============================================================
# Dual Encoder embedding extraction
# ============================================================
@torch.no_grad()
def encode_dataset(model, dataset, args, is_query):
    """
    Encode an entire dataset with the frozen dual encoder.
    Embeddings are kept on CPU to save GPU memory.
    """
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

        if is_query:
            vec = model(nl_inputs=inputs)
        else:
            vec = model(code_inputs=inputs)

        embeddings.append(vec.detach().cpu())

    return torch.cat(embeddings, dim=0)


# ============================================================
# Top-K retrieval
# ============================================================
@torch.no_grad()
def retrieve_topk(query_embeddings, code_embeddings, topk, args):
    """
    Retrieve top-k code candidates for every query.
    query_embeddings: [Nq, D]
    code_embeddings:  [Nc, D]
    Returns:
        topk_indices: [Nq, K]
        topk_scores:  [Nq, K]
    """
    query_embeddings = F.normalize(query_embeddings, dim=-1)
    code_embeddings = F.normalize(code_embeddings, dim=-1)

    all_indices = []
    all_scores = []

    for start in range(
        0, query_embeddings.size(0), args.retrieval_batch_size
    ):
        end = min(
            start + args.retrieval_batch_size,
            query_embeddings.size(0),
        )
        q = query_embeddings[start:end].to(args.device)
        c = code_embeddings.to(args.device)

        scores = torch.matmul(q, c.transpose(0, 1))
        k = min(topk, scores.size(1))
        values, indices = torch.topk(scores, k=k, dim=1)

        all_scores.append(values.cpu())
        all_indices.append(indices.cpu())

    return (
        torch.cat(all_indices, dim=0),
        torch.cat(all_scores, dim=0),
    )


# ============================================================
# Cross Encoder embedding extraction (NO gradient)
# ============================================================
@torch.no_grad()
def build_cross_embeddings(
    cross_encoder,
    query_dataset,
    code_dataset,
    topk_indices,
    args,
):
    """
    Build frozen cross-encoder representations for all retrieved pairs.
    Used for VALIDATION / EVAL / TEST only.
    Input:
        topk_indices: [Nq, K]
    Output:
        cross_embeddings: [Nq, K, D]
    Processing:
        cross_batch_size queries at a time.
        Each batch therefore contains:
            cross_batch_size * topk
        query-code pairs.
    """
    _, query_nl_ids, _ = dataset_tensors(query_dataset)
    code_code_ids, _, _ = dataset_tensors(code_dataset)

    num_queries = query_nl_ids.size(0)
    actual_topk = topk_indices.size(1)

    all_embeddings = []
    cross_encoder.eval()

    for q_start in range(0, num_queries, args.cross_batch_size):
        q_end = min(q_start + args.cross_batch_size, num_queries)

        query_batch = query_nl_ids[q_start:q_end]
        index_batch = topk_indices[q_start:q_end]

        batch_queries = []
        batch_codes = []
        current_q = query_batch.size(0)

        for i in range(current_q):
            q = query_batch[i]
            indices = index_batch[i]
            codes = code_code_ids[indices]

            q = q.unsqueeze(0).expand(actual_topk, -1)
            batch_queries.append(q)
            batch_codes.append(codes)

        batch_queries = torch.cat(batch_queries, dim=0).to(
            args.device, non_blocking=True
        )
        batch_codes = torch.cat(batch_codes, dim=0).to(
            args.device, non_blocking=True
        )

        embeddings = cross_encoder(
            query_ids=batch_queries,
            code_ids=batch_codes,
        )
        embeddings = embeddings.detach().cpu().view(
            current_q, actual_topk, -1
        )
        all_embeddings.append(embeddings)

        logger.info(
            "Cross encoding queries %d-%d / %d",
            q_start,
            q_end,
            num_queries,
        )

    return torch.cat(all_embeddings, dim=0)


# ============================================================
# Labels
# ============================================================
def build_labels(query_dataset, code_dataset, topk_indices):
    """
    Binary relevance labels.
    Positive: candidate code URL == query URL.
    """
    query_urls = [e.url for e in query_dataset.examples]
    code_urls = [e.url for e in code_dataset.examples]

    labels = torch.zeros(topk_indices.size(), dtype=torch.float32)
    for i, query_url in enumerate(query_urls):
        for j in range(topk_indices.size(1)):
            code_idx = int(topk_indices[i, j])
            if code_urls[code_idx] == query_url:
                labels[i, j] = 1.0
    return labels


# ============================================================
# Joint training of Cross Encoder + MLP
# ============================================================
def train_cross_reranker(
    args, cross_encoder, mlp, train_cache, labels, validation_cache
):
    """
    Jointly train CrossEncoder + MLP. The dual-encoder top-k is fixed.

    train_cache      = (query_dataset, code_dataset, topk_indices)
    validation_cache = (query_dataset, code_dataset, topk_indices)

    Cross embeddings are NEVER pre-cached for the training split because
    the cross encoder weights change every step.
    """
    query_dataset, code_dataset, topk_indices = train_cache
    _, query_nl_ids, _ = dataset_tensors(query_dataset)
    code_ids, _, _ = dataset_tensors(code_dataset)

    topk_indices = topk_indices.cpu()
    labels = labels.float().cpu()

    dataset = TensorDataset(torch.arange(len(query_dataset)), labels)
    dataloader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset),
        batch_size=args.mlp_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Both cross encoder and MLP receive gradients.
    params = list(cross_encoder.parameters()) + list(mlp.parameters())
    optimizer = AdamW(
        params,
        lr=args.learning_rate,
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    accum = max(1, args.gradient_accumulation_steps)
    logger.info(
        "Training: mlp_batch_size=%d, topk=%d, effective_query_batch=%d, "
        "grad_accum=%d",
        args.mlp_batch_size,
        topk_indices.size(1),
        args.mlp_batch_size * accum,
        accum,
    )

    best_mrr = float("-inf")

    for epoch in range(1, args.num_train_epochs + 1):
        cross_encoder.train()
        mlp.train()
        total_loss = 0.0
        total_steps = 0
        optimizer.zero_grad()

        for step, (query_indices, y) in enumerate(dataloader):
            query_indices = query_indices.long()
            y = y.to(args.device, non_blocking=True)

            batch_topk = topk_indices[query_indices]
            batch_size, k = batch_topk.shape

            q = query_nl_ids[query_indices].to(args.device)
            c = code_ids[batch_topk.reshape(-1)].to(args.device)
            q = q.unsqueeze(1).expand(-1, k, -1).reshape(batch_size * k, -1)

            # Grad flows through cross encoder into MLP.
            cross_features = cross_encoder(q, c)
            scores = mlp(cross_features).view(batch_size, k)

            loss = adaptive_hard_rerank_loss(
                scores=scores,
                labels=y,
                temperature=args.temperature,
                margin_scale=args.margin_scale,
                hard_k=args.hard_k,
            )

            (loss / accum).backward()

            if (step + 1) % accum == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            total_steps += 1
            if (step + 1) % args.log_steps == 0:
                logger.info(
                    "epoch %d step %d loss %.6f",
                    epoch,
                    step + 1,
                    total_loss / total_steps,
                )

        logger.info(
            "Epoch %d loss %.6f",
            epoch,
            total_loss / max(total_steps, 1),
        )

        # --------------------------------------------------
        # Validation: recompute cross embeddings with current weights.
        # --------------------------------------------------
        valid_q, valid_code, valid_topk = validation_cache
        cross_encoder.eval()
        valid_embeddings = build_cross_embeddings(
            cross_encoder, valid_q, valid_code, valid_topk, args
        )
        cross_encoder.train()

        valid_result = evaluate_cached(
            args, mlp, valid_q, valid_code, valid_topk, valid_embeddings
        )
        logger.info(
            "Epoch %d validation MRR %.6f, top-k recall %.6f",
            epoch,
            valid_result["mrr"],
            valid_result["topk_recall"],
        )

        if valid_result["mrr"] > best_mrr:
            best_mrr = valid_result["mrr"]
            save_best_joint(args, cross_encoder, mlp)


# ============================================================
# Cached evaluation (MLP only)
# ============================================================
@torch.no_grad()
def evaluate_cached(
    args,
    mlp,
    query_dataset,
    code_dataset,
    topk_indices,
    cross_embeddings,
):
    """
    Evaluate using already computed cross-encoder embeddings.
    No dual/cross encoder computation is performed here.
    """
    mlp.eval()
    scores = []

    for start in range(
        0, cross_embeddings.size(0), args.mlp_inference_batch_size
    ):
        end = min(
            start + args.mlp_inference_batch_size,
            cross_embeddings.size(0),
        )
        x = cross_embeddings[start:end]
        x = x.reshape(-1, x.size(-1)).to(args.device, non_blocking=True)

        logits = mlp(x)
        logits = logits.view(end - start, -1)
        scores.append(logits.cpu())

    scores = torch.cat(scores, dim=0)

    query_urls = [e.url for e in query_dataset.examples]
    code_urls = [e.url for e in code_dataset.examples]

    ranks = []
    topk_hits = []

    for i, query_url in enumerate(query_urls):
        order = torch.argsort(scores[i], descending=True)
        rank = None
        for position, candidate_position in enumerate(
            order.tolist(), start=1
        ):
            code_idx = int(topk_indices[i, candidate_position])
            if code_urls[code_idx] == query_url:
                rank = position
                break

        if rank is None:
            ranks.append(0.0)
        else:
            ranks.append(1.0 / rank)

        hit = any(
            code_urls[int(x)] == query_url for x in topk_indices[i]
        )
        topk_hits.append(float(hit))

    return {
        "mrr": float(np.mean(ranks)),
        "topk_recall": float(np.mean(topk_hits)),
    }


# ============================================================
# Model preparation
# ============================================================
def prepare_models(args, tokenizer):
    """
    Build dual encoder (frozen), cross encoder (trainable), MLP (trainable).
    """
    # --------------------------------------------------------
    # Dual encoder
    # --------------------------------------------------------
    dual_model_path = args.model_name_or_path
    if not args.do_zero_shot and args.dual_checkpoint is not None:
        dual_model_path = args.dual_checkpoint

    dual_backbone = RobertaModel.from_pretrained(dual_model_path)
    dual_model = Model(dual_backbone, args)

    # --------------------------------------------------------
    # Cross encoder
    # --------------------------------------------------------
    cross_model_path = (
        args.cross_model_name_or_path
        if args.cross_model_name_or_path
        else args.model_name_or_path
    )
    if not args.do_zero_shot and args.cross_checkpoint is not None:
        cross_model_path = args.cross_checkpoint

    cross_backbone = RobertaModel.from_pretrained(cross_model_path)
    cross_encoder = CrossEncoder(cross_backbone, tokenizer)
    cross_encoder.max_length = args.cross_length

    # --------------------------------------------------------
    # Freeze ONLY the dual encoder.
    # Cross encoder is intentionally left trainable.
    # --------------------------------------------------------
    for parameter in dual_model.parameters():
        parameter.requires_grad = False
    dual_model.eval()

    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------
    hidden_size = cross_backbone.config.hidden_size
    mlp = MLPReranker(
        input_dim=hidden_size,
        hidden_dims=args.mlp_hidden_dims,
        dropout=args.mlp_dropout,
    )

    if args.mlp_checkpoint is not None and os.path.exists(
        args.mlp_checkpoint
    ):
        load_mlp_checkpoint(mlp, args.mlp_checkpoint, args.device)

    dual_model.to(args.device)
    cross_encoder.to(args.device)
    mlp.to(args.device)

    # --------------------------------------------------------
    # DataParallel
    # --------------------------------------------------------
    if args.n_gpu > 1:
        logger.info("Using DataParallel with %d GPUs.", args.n_gpu)
        dual_model = nn.DataParallel(dual_model)
        cross_encoder = nn.DataParallel(cross_encoder)
        mlp = nn.DataParallel(mlp)

    return dual_model, cross_encoder, mlp


# ============================================================
# Cache preparation (dual embeddings + top-K only)
# ============================================================
def prepare_rerank_cache(
    args,
    tokenizer,
    dual_model,
    query_file,
    code_file,
    is_train=False,
):
    """
    Prepare dual-encoder embeddings + top-K indices.

    Cross-encoder embeddings are intentionally NOT computed here:
      - during training they must be recomputed every step;
      - during validation they must be recomputed every epoch;
      - during eval/test they are computed once by the caller.
    """
    query_dataset, code_dataset = build_dataset(
        tokenizer, args, query_file, code_file, is_train=is_train,
    )
    logger.info("Queries: %d", len(query_dataset))
    logger.info("Codes: %d", len(code_dataset))

    # Dual encoder embeddings (frozen)
    query_embeddings = encode_dataset(
        dual_model, query_dataset, args, is_query=True
    )
    code_embeddings = encode_dataset(
        dual_model, code_dataset, args, is_query=False
    )

    # Top-K retrieval
    topk_indices, _ = retrieve_topk(
        query_embeddings, code_embeddings, args.topk, args
    )
    logger.info(
        "Retrieved top-%d candidates for %d queries.",
        topk_indices.size(1),
        topk_indices.size(0),
    )

    return query_dataset, code_dataset, topk_indices


# ============================================================
# Stage runner
# ============================================================
def run_stage(
    args,
    tokenizer,
    dual_model,
    cross_encoder,
    mlp,
    query_file,
    code_file,
    is_train,
    validation_file=None,
    validation_code_file=None,
):
    """
    Train or evaluate one stage.
    """
    if is_train and (validation_file is None or validation_code_file is None):
        raise ValueError(
            "Validation query and codebase files are required "
            "to select the MLP checkpoint by validation MRR."
        )

    cache = prepare_rerank_cache(
        args,
        tokenizer,
        dual_model,
        query_file,
        code_file,
        is_train=is_train,
    )

    if is_train:
        validation_cache = prepare_rerank_cache(
            args,
            tokenizer,
            dual_model,
            validation_file,
            validation_code_file,
            is_train=False,
        )
        labels = build_labels(cache[0], cache[1], cache[2])

        train_cross_reranker(
            args,
            cross_encoder,
            mlp,
            cache,
            labels,
            validation_cache,
        )
        return None

    # --------------------------------------------------------
    # Eval / Test: compute cross embeddings ONCE and evaluate.
    # --------------------------------------------------------
    query_dataset, code_dataset, topk_indices = cache
    cross_embeddings = build_cross_embeddings(
        cross_encoder, query_dataset, code_dataset, topk_indices, args
    )
    logger.info(
        "Cross embeddings shape: %s",
        tuple(cross_embeddings.shape),
    )
    return evaluate_cached(
        args, mlp, query_dataset, code_dataset, topk_indices, cross_embeddings
    )


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

    # Models
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="HuggingFace checkpoint for the dual encoder.",
    )
    parser.add_argument(
        "--dual_checkpoint",
        type=str,
        default=None,
        help="Optional Hugging Face directory for a trained dual encoder.",
    )
    parser.add_argument(
        "--cross_model_name_or_path",
        type=str,
        default=None,
        help="Optional HuggingFace checkpoint for the cross encoder.",
    )
    parser.add_argument(
        "--cross_checkpoint",
        type=str,
        default=None,
        help="Optional Hugging Face directory for a trained cross encoder.",
    )
    parser.add_argument(
        "--mlp_checkpoint",
        type=str,
        default=None,
        help="Optional trained MLP checkpoint.",
    )

    # Modes
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument(
        "--do_zero_shot",
        action="store_true",
        help=(
            "Use HuggingFace dual/cross encoders directly "
            "instead of loading trained checkpoints."
        ),
    )

    # Sequence lengths
    parser.add_argument("--nl_length", type=int, default=128)
    parser.add_argument("--code_length", type=int, default=256)
    parser.add_argument(
        "--cross_length",
        type=int,
        default=384,
        help="Maximum sequence length of [CLS] query [SEP] code [SEP].",
    )

    # Retrieval
    parser.add_argument(
        "--topk",
        type=int,
        default=100,
        help="Number of candidates retrieved by dual encoder.",
    )
    parser.add_argument(
        "--dual_batch_size",
        type=int,
        default=32,
        help="Batch size for dual encoder embedding.",
    )
    parser.add_argument(
        "--retrieval_batch_size",
        type=int,
        default=32,
        help="Number of queries per dual retrieval batch.",
    )

    # Cross encoder (for validation/eval feature extraction)
    parser.add_argument(
        "--cross_batch_size",
        type=int,
        default=4,
        help=(
            "Number of queries per frozen cross-encoder batch "
            "(validation/eval/test). Actual pair batch size is "
            "cross_batch_size * topk."
        ),
    )

    # MLP
    parser.add_argument(
        "--mlp_hidden_dims", nargs="+", type=int, default=[512]
    )
    parser.add_argument("--mlp_dropout", type=float, default=0.1)
    parser.add_argument(
        "--mlp_batch_size",
        type=int,
        default=16,
        help=(
            "Number of queries per training step. Actual pair batch size "
            "is mlp_batch_size * topk. Keep this small to avoid OOM "
            "when the cross encoder receives gradients."
        ),
    )
    parser.add_argument(
        "--mlp_inference_batch_size", type=int, default=256
    )

    # Adaptive hard-negative ranking loss
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--margin_scale", type=float, default=0.2)
    parser.add_argument("--hard_k", type=int, default=2)

    # Training
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=(
            "Accumulate gradients over this many mlp_batch_size steps "
            "before calling optimizer.step(). Effective query batch is "
            "mlp_batch_size * gradient_accumulation_steps."
        ),
    )

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
            raise ValueError(
                "--eval_data_file is required for per-epoch validation "
                "when using --do_train"
            )
        if args.codebase_file is None:
            raise ValueError(
                "--codebase_file is required for validation MRR "
                "when using --do_train"
            )
        if args.temperature <= 0:
            raise ValueError("--temperature must be greater than 0")
        if args.hard_k <= 0:
            raise ValueError("--hard_k must be greater than 0")
        if args.gradient_accumulation_steps < 1:
            raise ValueError(
                "--gradient_accumulation_steps must be >= 1"
            )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info("Arguments:\n%s", json.dumps(vars(args), indent=4))

    # Device
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    args.n_gpu = torch.cuda.device_count()
    args.frozen_layers = (
        args.frozen_layers.split(",") if args.frozen_layers else []
    )
    logger.info("device: %s, n_gpu: %d", args.device, args.n_gpu)

    set_seed(args.seed)

    # Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)

    # Models
    dual_model, cross_encoder, mlp = prepare_models(args, tokenizer)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    if args.do_train:
        run_stage(
            args=args,
            tokenizer=tokenizer,
            dual_model=dual_model,
            cross_encoder=cross_encoder,
            mlp=mlp,
            query_file=args.train_data_file,
            code_file=args.train_data_file,
            is_train=True,
            validation_file=args.eval_data_file,
            validation_code_file=args.codebase_file,
        )

    # --------------------------------------------------------
    # Load the best checkpoints after training
    # --------------------------------------------------------
    if args.do_train:
        args.mlp_checkpoint = best_mlp_checkpoint(args)
    elif args.mlp_checkpoint is None and (args.do_eval or args.do_test):
        args.mlp_checkpoint = best_mlp_checkpoint(args)

    if args.do_train:
        load_state_dict(
            unwrap_model(cross_encoder),
            best_cross_checkpoint(args),
            args.device,
        )

    if args.do_train or args.do_eval or args.do_test:
        load_mlp_checkpoint(mlp, args.mlp_checkpoint, args.device)

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    if args.do_eval:
        if args.eval_data_file is None:
            raise ValueError("--eval_data_file is required for --do_eval")
        if args.codebase_file is None:
            raise ValueError("--codebase_file is required for --do_eval")

        result = run_stage(
            args=args,
            tokenizer=tokenizer,
            dual_model=dual_model,
            cross_encoder=cross_encoder,
            mlp=mlp,
            query_file=args.eval_data_file,
            code_file=args.codebase_file,
            is_train=False,
        )
        logger.info("***** Eval results *****")
        assert result is not None
        for key, value in result.items():
            logger.info("  %s = %.6f", key, value)

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------
    if args.do_test:
        if args.test_data_file is None:
            raise ValueError("--test_data_file is required for --do_test")
        if args.codebase_file is None:
            raise ValueError("--codebase_file is required for --do_test")

        result = run_stage(
            args=args,
            tokenizer=tokenizer,
            dual_model=dual_model,
            cross_encoder=cross_encoder,
            mlp=mlp,
            query_file=args.test_data_file,
            code_file=args.codebase_file,
            is_train=False,
        )
        logger.info("***** Test results *****")
        assert result is not None
        for key, value in result.items():
            logger.info("  %s = %.6f", key, value)


if __name__ == "__main__":
    main()