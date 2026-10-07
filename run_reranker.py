# coding=utf-8
"""
Dual Encoder -> Top-K -> Cross Encoder -> MLP Reranker

Pipeline:
    1. Load a dual encoder:
         - from --dual_checkpoint, or
         - directly from HuggingFace with --do_zero_shot
    2. Encode all queries and all candidate codes.
    3. Retrieve top-k candidates using dual-encoder similarity.
    4. Run a frozen cross encoder on all query/top-k-code pairs.
    5. Cache cross-encoder embeddings.
    6. Train ONLY the MLP reranker using cached embeddings.
    7. Evaluate/Test with the same frozen dual + cross encoders.

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
    - Cross encoder is frozen.
    - Only MLP parameters are trainable.
    - Dual retrieval and cross-encoder representations are computed under
      torch.no_grad().
    - cross_batch_size is the number of queries processed at once.
      Therefore each batch contains:
          cross_batch_size * topk
      query-code pairs.
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

    state_dict = torch.load(
        checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    logger.info("Checkpoint loaded.")


def best_mlp_checkpoint(args):
    return os.path.join(
        args.output_dir,
        "checkpoint-best-mrr",
        "mlp.bin",
    )


def save_best_mlp(args, mlp):
    checkpoint = best_mlp_checkpoint(args)
    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)

    torch.save(
        unwrap_model(mlp).state_dict(),
        checkpoint,
    )

    logger.info(
        "Saved best validation-MRR MLP checkpoint to %s",
        checkpoint,
    )


def load_mlp_checkpoint(mlp, checkpoint, device):
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            "MLP checkpoint not found: %s" % checkpoint
        )

    load_state_dict(
        unwrap_model(mlp),
        checkpoint,
        device,
    )


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
        scores:
            [B, K] MLP scores.

        labels:
            [B, K], where 1 denotes the positive candidate.

        temperature:
            Controls hard-negative weighting and ranking
            loss sharpness.

        margin_scale:
            Scale of the adaptive margin.

        hard_k:
            Number of hardest negatives used per query.
    """

    B, K = scores.shape

    positive_mask = labels.bool()

    # ---------------------------------------------------------
    # Keep only queries that contain a positive candidate.
    # ---------------------------------------------------------

    valid = positive_mask.any(dim=1)

    if not valid.any():
        return scores.sum() * 0.0

    scores = scores[valid]
    positive_mask = positive_mask[valid]

    # ---------------------------------------------------------
    # Positive score
    #
    # Expected shape:
    # [B_valid]
    #
    # This assumes one positive per query.
    # ---------------------------------------------------------

    pos = scores.masked_select(
        positive_mask
    )

    # ---------------------------------------------------------
    # Negative scores
    #
    # [B_valid, K]
    #
    # Positive candidate is masked out.
    # ---------------------------------------------------------

    neg = scores.masked_fill(
        positive_mask,
        -1e9,
    )

    # ---------------------------------------------------------
    # Select hardest negatives.
    # ---------------------------------------------------------

    num_neg = K - 1

    k = min(
        hard_k,
        num_neg,
    )

    hard_neg, _ = torch.topk(
        neg,
        k=k,
        dim=1,
    )

    # ---------------------------------------------------------
    # Adaptive margin.
    #
    # MLP output is an arbitrary logit, so do NOT use:
    #
    #     margin_scale * (1 - pos)
    #
    # Instead map the positive score to [0, 1].
    #
    # Low positive confidence
    #     -> larger margin
    #
    # High positive confidence
    #     -> smaller margin
    # ---------------------------------------------------------

    margin = (
        margin_scale
        * (
            1.0
            - torch.sigmoid(pos)
        )
    )

    # ---------------------------------------------------------
    # Harder negatives receive larger weights.
    # ---------------------------------------------------------

    weights = F.softmax(
        hard_neg / temperature,
        dim=1,
    )

    # ---------------------------------------------------------
    # Ranking objective:
    #
    # hard_neg < pos - margin
    #
    # Equivalent violation:
    #
    # hard_neg - pos + margin
    # ---------------------------------------------------------

    ranking_loss = F.softplus(
        (
            hard_neg
            - pos.unsqueeze(1)
            + margin.unsqueeze(1)
        )
        / temperature
    )

    # ---------------------------------------------------------
    # Weighted hard-negative loss.
    # ---------------------------------------------------------

    loss = (
        weights
        * ranking_loss
    ).sum(dim=1)

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

    The model itself is frozen during reranker training.
    """

    def __init__(self, encoder, tokenizer):
        super().__init__()

        self.encoder = encoder
        self.tokenizer = tokenizer

        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def build_pair_inputs(
        self,
        query_ids,
        code_ids,
    ):
        """
        Construct:

            [CLS] query [SEP] code [SEP]

        from already padded query/code tensors.

        Returns:
            pair_ids:      [B, L]
            attention_mask:[B, L]
        """

        device = query_ids.device
        batch_size = query_ids.size(0)

        sequences = []

        for i in range(batch_size):
            query = query_ids[i]
            code = code_ids[i]

            # Remove padding.
            query = query[query.ne(self.pad_token_id)]
            code = code[code.ne(self.pad_token_id)]

            # Remove existing special tokens if possible.
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

            pair = torch.cat(
                [
                    torch.tensor(
                        [self.cls_token_id],
                        device=device,
                        dtype=torch.long,
                    ),
                    query,
                    torch.tensor(
                        [self.sep_token_id],
                        device=device,
                        dtype=torch.long,
                    ),
                    code,
                    torch.tensor(
                        [self.sep_token_id],
                        device=device,
                        dtype=torch.long,
                    ),
                ]
            )

            sequences.append(pair)

        max_length = max(
            x.size(0)
            for x in sequences
        )

        # The original dual encoder uses:
        #   nl_length = 128
        #   code_length = 256
        #
        # For the cross encoder we allow:
        #   cross_length
        max_length = min(
            max_length,
            self.max_length,
        )

        pair_ids = torch.full(
            (batch_size, max_length),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )

        for i, seq in enumerate(sequences):
            seq = seq[:max_length]
            pair_ids[i, :seq.size(0)] = seq

        attention_mask = pair_ids.ne(
            self.pad_token_id
        ).long()

        return pair_ids, attention_mask

    def forward(
        self,
        query_ids,
        code_ids,
    ):
        pair_ids, attention_mask = self.build_pair_inputs(
            query_ids,
            code_ids,
        )

        outputs = self.encoder(
            pair_ids,
            attention_mask=attention_mask,
        )

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

    Input:
        cross-encoder embedding [B, D]

    Output:
        logits [B]

    Only this module is optimized.
    """

    def __init__(
        self,
        input_dim,
        hidden_dims=[512],
        dropout=0.1,
    ):
        super().__init__()

        layers = []
        for i, hidden_dim in enumerate(hidden_dims):
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

def build_dataset(
    tokenizer,
    args,
    query_file,
    code_file=None,
    is_train=False,
):
    """
    Build query/code datasets.

    Train:
        query_file == train file
        code_file  == train file

    Eval/Test:
        query_file == eval/test file
        code_file  == codebase file
    """

    query_dataset = TextDataset(
        tokenizer,
        args,
        query_file,
    )

    if is_train:
        code_dataset = TextDataset(
            tokenizer,
            args,
            query_file,
        )
    else:
        code_dataset = TextDataset(
            tokenizer,
            args,
            code_file,
        )

    return query_dataset, code_dataset


def dataset_tensors(dataset):
    """
    Extract tensors from TextDataset.

    Returns:
        code_ids: [N, code_length]
        nl_ids:   [N, nl_length]
        urls: list[str]
    """

    code_ids = torch.tensor(
        [x.code_ids for x in dataset.examples],
        dtype=torch.long,
    )

    nl_ids = torch.tensor(
        [x.nl_ids for x in dataset.examples],
        dtype=torch.long,
    )

    urls = [
        x.url
        for x in dataset.examples
    ]

    return code_ids, nl_ids, urls


# ============================================================
# Dual Encoder
# ============================================================

@torch.no_grad()
def encode_dataset(
    model,
    dataset,
    args,
    is_query,
):
    """
    Encode an entire dataset with the frozen dual encoder.

    Returned embeddings are kept on CPU to avoid unnecessarily occupying
    GPU memory.
    """

    sampler = torch.utils.data.SequentialSampler(
        dataset
    )

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

        if is_query:
            inputs = batch[1]
        else:
            inputs = batch[0]

        inputs = inputs.to(
            args.device,
            non_blocking=True,
        )

        if is_query:
            vec = model(
                nl_inputs=inputs
            )
        else:
            vec = model(
                code_inputs=inputs
            )

        embeddings.append(
            vec.detach().cpu()
        )

    return torch.cat(
        embeddings,
        dim=0,
    )


# ============================================================
# Top-K retrieval
# ============================================================

@torch.no_grad()
def retrieve_topk(
    query_embeddings,
    code_embeddings,
    topk,
    args,
):
    """
    Retrieve top-k code candidates for every query.

    query_embeddings:
        [Nq, D]

    code_embeddings:
        [Nc, D]

    Returns:
        topk_indices:
            [Nq, K]

        topk_scores:
            [Nq, K]
    """

    query_embeddings = F.normalize(
        query_embeddings,
        dim=-1,
    )

    code_embeddings = F.normalize(
        code_embeddings,
        dim=-1,
    )

    all_indices = []
    all_scores = []

    # Compute in chunks so that Nq x Nc similarity matrix does not
    # necessarily have to exist entirely on GPU.
    for start in range(
        0,
        query_embeddings.size(0),
        args.retrieval_batch_size,
    ):

        end = min(
            start + args.retrieval_batch_size,
            query_embeddings.size(0),
        )

        q = query_embeddings[
            start:end
        ].to(args.device)

        c = code_embeddings.to(args.device)

        scores = torch.matmul(
            q,
            c.transpose(0, 1),
        )

        k = min(
            topk,
            scores.size(1),
        )

        values, indices = torch.topk(
            scores,
            k=k,
            dim=1,
        )

        all_scores.append(
            values.cpu()
        )

        all_indices.append(
            indices.cpu()
        )

    return (
        torch.cat(all_indices, dim=0),
        torch.cat(all_scores, dim=0),
    )


# ============================================================
# Cross Encoder embedding extraction
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

    query_code_ids, query_nl_ids, query_urls = dataset_tensors(
        query_dataset
    )

    code_code_ids, _, code_urls = dataset_tensors(
        code_dataset
    )

    num_queries = query_nl_ids.size(0)
    actual_topk = topk_indices.size(1)

    # Infer cross representation dimension with the first batch.
    all_embeddings = []

    cross_encoder.eval()

    for q_start in range(
        0,
        num_queries,
        args.cross_batch_size,
    ):

        q_end = min(
            q_start + args.cross_batch_size,
            num_queries,
        )

        query_batch = query_nl_ids[
            q_start:q_end
        ]

        index_batch = topk_indices[
            q_start:q_end
        ]

        batch_queries = []
        batch_codes = []

        current_q = query_batch.size(0)

        for i in range(current_q):

            q = query_batch[i]

            indices = index_batch[i]

            codes = code_code_ids[
                indices
            ]

            q = q.unsqueeze(0).expand(
                actual_topk,
                -1,
            )

            batch_queries.append(q)
            batch_codes.append(codes)

        batch_queries = torch.cat(
            batch_queries,
            dim=0,
        )

        batch_codes = torch.cat(
            batch_codes,
            dim=0,
        )

        batch_queries = batch_queries.to(
            args.device,
            non_blocking=True,
        )

        batch_codes = batch_codes.to(
            args.device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Cross encoder
        #
        # Number of pairs:
        #     current_q * topk
        # ----------------------------------------------------
        embeddings = cross_encoder(
            query_ids=batch_queries,
            code_ids=batch_codes,
        )

        embeddings = embeddings.detach().cpu()

        embeddings = embeddings.view(
            current_q,
            actual_topk,
            -1,
        )

        all_embeddings.append(
            embeddings
        )

        logger.info(
            "Cross encoding queries %d-%d / %d",
            q_start,
            q_end,
            num_queries,
        )

    return torch.cat(
        all_embeddings,
        dim=0,
    )


# ============================================================
# Labels
# ============================================================

def build_labels(
    query_dataset,
    code_dataset,
    topk_indices,
):
    """
    Construct binary relevance labels.

    Positive:
        candidate code URL == query URL

    Negative:
        otherwise.
    """

    query_urls = [
        example.url
        for example in query_dataset.examples
    ]

    code_urls = [
        example.url
        for example in code_dataset.examples
    ]

    labels = torch.zeros(
        topk_indices.size(),
        dtype=torch.float32,
    )

    for i, query_url in enumerate(query_urls):

        for j in range(
            topk_indices.size(1)
        ):

            code_idx = int(
                topk_indices[i, j]
            )

            if code_urls[code_idx] == query_url:
                labels[i, j] = 1.0

    return labels


# ============================================================
# MLP training
# ============================================================

def train_mlp(
    args,
    mlp,
    cross_embeddings,
    labels,
    validation_cache,
):
    """
    Train ONLY the MLP with adaptive hard-negative ranking loss.

    cross_embeddings:
        [Nq, K, D]

    labels:
        [Nq, K]

    For each query:
        - one positive candidate
        - K-1 negative candidates

    The MLP produces one score per candidate:
        [B, K]

    The adaptive hard-negative loss then selects the
    hardest negatives within each query's candidate set.
    """

    # ---------------------------------------------------------
    # Keep query dimension.
    #
    # embeddings: [Nq, K, D]
    # labels:     [Nq, K]
    # ---------------------------------------------------------

    dataset = TensorDataset(
        cross_embeddings,
        labels,
    )

    sampler = RandomSampler(
        dataset
    )

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.mlp_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = AdamW(
        mlp.parameters(),
        lr=args.learning_rate,
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    num_queries = cross_embeddings.size(0)
    topk = cross_embeddings.size(1)
    hidden_dim = cross_embeddings.size(2)

    num_positive = int(
        labels.sum().item()
    )

    num_negative = int(
        (labels == 0).sum().item()
    )

    num_valid_queries = int(
        (labels.sum(dim=1) > 0).sum().item()
    )

    logger.info(
        "***** Training MLP *****"
    )

    logger.info(
        "  Num queries = %d",
        num_queries,
    )

    logger.info(
        "  Top-K = %d",
        topk,
    )

    logger.info(
        "  Cross embedding dim = %d",
        hidden_dim,
    )

    logger.info(
        "  Positive pairs = %d",
        num_positive,
    )

    logger.info(
        "  Negative pairs = %d",
        num_negative,
    )

    logger.info(
        "  Queries with positive = %d / %d",
        num_valid_queries,
        num_queries,
    )

    # ---------------------------------------------------------
    best_mrr = float("-inf")

    for epoch in range(
        args.num_train_epochs
    ):

        mlp.train()

        total_loss = 0.0
        total_steps = 0

        for step, batch in enumerate(
            dataloader
        ):

            # -------------------------------------------------
            # x:
            #   [B, K, D]
            #
            # y:
            #   [B, K]
            # -------------------------------------------------

            x = batch[0].to(
                args.device,
                non_blocking=True,
            )

            y = batch[1].to(
                args.device,
                non_blocking=True,
            )

            batch_size = x.size(0)
            k = x.size(1)
            dim = x.size(2)

            # -------------------------------------------------
            # MLP normally accepts [N, D].
            #
            # Flatten only for the MLP forward pass:
            #
            # [B, K, D]
            #       ↓
            # [B*K, D]
            #       ↓
            # [B*K, 1]
            #       ↓
            # [B, K]
            # -------------------------------------------------

            x_flat = x.reshape(
                batch_size * k,
                dim,
            )

            logits = mlp(
                x_flat
            )

            scores = logits.view(
                batch_size,
                k,
            )

            # -------------------------------------------------
            # Adaptive hard-negative ranking loss
            #
            # IMPORTANT:
            # y must remain [B, K].
            #
            # The loss selects hard negatives independently
            # for every query.
            # -------------------------------------------------

            loss = adaptive_hard_rerank_loss(
                scores=scores,
                labels=y,
                temperature=args.temperature,
                margin_scale=args.margin_scale,
                hard_k=args.hard_k,
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                mlp.parameters(),
                args.max_grad_norm,
            )

            optimizer.step()

            total_loss += loss.item()
            total_steps += 1

            if (
                step + 1
            ) % args.log_steps == 0:

                logger.info(
                    "epoch %d step %d loss %.6f",
                    epoch,
                    step + 1,
                    total_loss / total_steps,
                )

        epoch_loss = (
            total_loss
            / max(total_steps, 1)
        )

        logger.info(
            "Epoch %d loss %.6f",
            epoch,
            epoch_loss,
        )

        valid_result = evaluate_cached(
            args,
            mlp,
            *validation_cache,
        )

        logger.info(
            "Epoch %d validation MRR %.6f, top-k recall %.6f",
            epoch,
            valid_result["mrr"],
            valid_result["topk_recall"],
        )

        # -----------------------------------------------------
        # Save the checkpoint with the best validation MRR.
        # -----------------------------------------------------

        if valid_result["mrr"] > best_mrr:
            best_mrr = valid_result["mrr"]
            save_best_mlp(
                args,
                mlp,
            )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def rerank_and_evaluate(
    args,
    mlp,
    query_dataset,
    code_dataset,
    dual_model,
    cross_encoder,
):
    """
    Complete inference:

        dual encoder
            -> top-k
            -> cross encoder
            -> MLP
            -> MRR
    """

    logger.info(
        "Encoding queries with dual encoder..."
    )

    query_embeddings = encode_dataset(
        dual_model,
        query_dataset,
        args,
        is_query=True,
    )

    logger.info(
        "Encoding codebase with dual encoder..."
    )

    code_embeddings = encode_dataset(
        dual_model,
        code_dataset,
        args,
        is_query=False,
    )

    topk_indices, dual_scores = retrieve_topk(
        query_embeddings,
        code_embeddings,
        args.topk,
        args,
    )

    logger.info(
        "Top-k retrieval complete."
    )

    cross_embeddings = build_cross_embeddings(
        cross_encoder,
        query_dataset,
        code_dataset,
        topk_indices,
        args,
    )

    mlp.eval()

    num_queries = cross_embeddings.size(0)

    rerank_scores = []

    for start in range(
        0,
        num_queries,
        args.mlp_inference_batch_size,
    ):

        end = min(
            start + args.mlp_inference_batch_size,
            num_queries,
        )

        x = cross_embeddings[
            start:end
        ].reshape(
            -1,
            cross_embeddings.size(-1),
        )

        logits = mlp(
            x.to(
                args.device,
                non_blocking=True,
            )
        )

        logits = logits.view(
            end - start,
            -1,
        )

        rerank_scores.append(
            logits.cpu()
        )

    rerank_scores = torch.cat(
        rerank_scores,
        dim=0,
    )

    query_urls = [
        example.url
        for example in query_dataset.examples
    ]

    code_urls = [
        example.url
        for example in code_dataset.examples
    ]

    ranks = []

    for i, query_url in enumerate(
        query_urls
    ):

        scores = rerank_scores[i]

        order = torch.argsort(
            scores,
            descending=True,
        )

        rank = None

        for position, candidate_position in enumerate(
            order.tolist(),
            start=1,
        ):

            code_idx = int(
                topk_indices[i, candidate_position]
            )

            if code_urls[code_idx] == query_url:
                rank = position
                break

        if rank is None:
            ranks.append(0.0)
        else:
            ranks.append(
                1.0 / rank
            )

    mrr = float(
        np.mean(ranks)
    )

    return {
        "mrr": mrr,
        "dual_topk_recall": float(
            np.mean(
                [
                    any(
                        code_urls[
                            int(idx)
                        ] == query_urls[i]
                        for idx in topk_indices[i]
                    )
                    for i in range(
                        len(query_urls)
                    )
                ]
            )
        ),
    }


# ============================================================
# Main pipeline
# ============================================================

def prepare_models(args, tokenizer):
    """
    Build dual encoder, cross encoder and MLP.

    Dual encoder:
        Model(RobertaModel(...), args)

    Cross encoder:
        CrossEncoder(RobertaModel(...), tokenizer)

    MLP:
        input_dim = cross encoder hidden size
    """

    # --------------------------------------------------------
    # Dual encoder
    # --------------------------------------------------------

    dual_model_path = args.model_name_or_path
    if (
        not args.do_zero_shot
        and args.dual_checkpoint is not None
    ):
        dual_model_path = args.dual_checkpoint

    dual_backbone = RobertaModel.from_pretrained(
        dual_model_path
    )

    dual_model = Model(
        dual_backbone,
        args,
    )

    # --------------------------------------------------------
    # Cross encoder
    # --------------------------------------------------------

    cross_model_path = (
        args.cross_model_name_or_path
        if args.cross_model_name_or_path
        else args.model_name_or_path
    )
    if (
        not args.do_zero_shot
        and args.cross_checkpoint is not None
    ):
        cross_model_path = args.cross_checkpoint

    cross_backbone = RobertaModel.from_pretrained(
        cross_model_path
    )

    cross_encoder = CrossEncoder(
        cross_backbone,
        tokenizer,
    )

    cross_encoder.max_length = args.cross_length

    # --------------------------------------------------------
    # Freeze dual and cross encoders
    # --------------------------------------------------------

    for parameter in dual_model.parameters():
        parameter.requires_grad = False

    for parameter in cross_encoder.parameters():
        parameter.requires_grad = False

    dual_model.eval()
    cross_encoder.eval()

    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------

    hidden_size = cross_backbone.config.hidden_size

    mlp = MLPReranker(
        input_dim=hidden_size,
        hidden_dims=args.mlp_hidden_dims,
        dropout=args.mlp_dropout,
    )

    if (
        args.mlp_checkpoint is not None
        and os.path.exists(args.mlp_checkpoint)
    ):
        load_mlp_checkpoint(
            mlp,
            args.mlp_checkpoint,
            args.device,
        )

    dual_model.to(args.device)
    cross_encoder.to(args.device)
    mlp.to(args.device)

    # --------------------------------------------------------
    # DataParallel
    # --------------------------------------------------------

    if args.n_gpu > 1:

        logger.info(
            "Using DataParallel with %d GPUs.",
            args.n_gpu,
        )

        dual_model = nn.DataParallel(
            dual_model
        )

        cross_encoder = nn.DataParallel(
            cross_encoder
        )

        mlp = nn.DataParallel(
            mlp
        )

    return (
        dual_model,
        cross_encoder,
        mlp,
    )


def prepare_rerank_cache(
    args,
    tokenizer,
    dual_model,
    cross_encoder,
    query_file,
    code_file,
    is_train=False,
):
    """
    Prepare frozen dual/cross-encoder features for one query/candidate set.
    """

    query_dataset, code_dataset = build_dataset(
        tokenizer,
        args,
        query_file,
        code_file,
        is_train=is_train,
    )

    logger.info(
        "Queries: %d",
        len(query_dataset),
    )

    logger.info(
        "Codes: %d",
        len(code_dataset),
    )

    # --------------------------------------------------------
    # Dual encoder embeddings
    # --------------------------------------------------------

    query_embeddings = encode_dataset(
        dual_model,
        query_dataset,
        args,
        is_query=True,
    )

    code_embeddings = encode_dataset(
        dual_model,
        code_dataset,
        args,
        is_query=False,
    )

    # --------------------------------------------------------
    # Top-k
    # --------------------------------------------------------

    topk_indices, topk_scores = retrieve_topk(
        query_embeddings,
        code_embeddings,
        args.topk,
        args,
    )

    logger.info(
        "Retrieved top-%d candidates for %d queries.",
        topk_indices.size(1),
        topk_indices.size(0),
    )

    # --------------------------------------------------------
    # Cross encoder
    # --------------------------------------------------------

    cross_embeddings = build_cross_embeddings(
        cross_encoder,
        query_dataset,
        code_dataset,
        topk_indices,
        args,
    )

    logger.info(
        "Cross embeddings shape: %s",
        tuple(cross_embeddings.shape),
    )

    return (
        query_dataset,
        code_dataset,
        topk_indices,
        cross_embeddings,
    )


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
    Execute feature preparation followed by MLP training or evaluation.
    """

    if is_train and (
        validation_file is None
        or validation_code_file is None
    ):
        raise ValueError(
            "Validation query and codebase files are required "
            "to select the MLP checkpoint by validation MRR."
        )

    cache = prepare_rerank_cache(
        args,
        tokenizer,
        dual_model,
        cross_encoder,
        query_file,
        code_file,
        is_train=is_train,
    )

    if is_train:
        validation_cache = prepare_rerank_cache(
            args,
            tokenizer,
            dual_model,
            cross_encoder,
            validation_file,
            validation_code_file,
        )

        labels = build_labels(
            cache[0],
            cache[1],
            cache[2],
        )

        train_mlp(
            args,
            mlp,
            cache[3],
            labels,
            validation_cache,
        )

    else:

        result = evaluate_cached(
            args,
            mlp,
            *cache,
        )

        return result

    return None


# ============================================================
# Cached evaluation
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
    Evaluate using already cached cross-encoder embeddings.

    No dual/cross encoder computation is performed here.
    """

    mlp.eval()

    scores = []

    for start in range(
        0,
        cross_embeddings.size(0),
        args.mlp_inference_batch_size,
    ):

        end = min(
            start + args.mlp_inference_batch_size,
            cross_embeddings.size(0),
        )

        x = cross_embeddings[
            start:end
        ]

        x = x.reshape(
            -1,
            x.size(-1),
        ).to(
            args.device,
            non_blocking=True,
        )

        logits = mlp(x)

        logits = logits.view(
            end - start,
            -1,
        )

        scores.append(
            logits.cpu()
        )

    scores = torch.cat(
        scores,
        dim=0,
    )

    query_urls = [
        example.url
        for example in query_dataset.examples
    ]

    code_urls = [
        example.url
        for example in code_dataset.examples
    ]

    ranks = []
    topk_hits = []

    for i, query_url in enumerate(
        query_urls
    ):

        order = torch.argsort(
            scores[i],
            descending=True,
        )

        rank = None

        for position, candidate_position in enumerate(
            order.tolist(),
            start=1,
        ):

            code_idx = int(
                topk_indices[
                    i,
                    candidate_position,
                ]
            )

            if code_urls[code_idx] == query_url:
                rank = position
                break

        if rank is None:
            ranks.append(0.0)
        else:
            ranks.append(
                1.0 / rank
            )

        # Whether the original positive is even present
        # in the dual-encoder top-k.
        hit = any(
            code_urls[
                int(x)
            ] == query_url
            for x in topk_indices[i]
        )

        topk_hits.append(
            float(hit)
        )

    return {
        "mrr": float(
            np.mean(ranks)
        ),
        "topk_recall": float(
            np.mean(topk_hits)
        ),
    }


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    parser.add_argument(
        "--train_data_file",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--eval_data_file",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--test_data_file",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--codebase_file",
        type=str,
        default=None,
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Modes
    # --------------------------------------------------------

    parser.add_argument(
        "--do_train",
        action="store_true",
    )

    parser.add_argument(
        "--do_eval",
        action="store_true",
    )

    parser.add_argument(
        "--do_test",
        action="store_true",
    )

    parser.add_argument(
        "--do_zero_shot",
        action="store_true",
        help=(
            "Use HuggingFace dual/cross encoders directly "
            "instead of loading trained checkpoints."
        ),
    )

    # --------------------------------------------------------
    # Sequence lengths
    # --------------------------------------------------------

    parser.add_argument(
        "--nl_length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--code_length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--cross_length",
        type=int,
        default=384,
        help=(
            "Maximum sequence length of "
            "[CLS] query [SEP] code [SEP]."
        ),
    )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Cross encoder
    # --------------------------------------------------------

    parser.add_argument(
        "--cross_batch_size",
        type=int,
        default=4,
        help=(
            "Number of queries per cross encoder batch. "
            "Actual pair batch size is "
            "cross_batch_size * topk."
        ),
    )

    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------

    parser.add_argument(
        "--mlp_hidden_dims",
        nargs="+",
        type=int,
        default=[512],
    )

    parser.add_argument(
        "--mlp_dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--mlp_batch_size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--mlp_inference_batch_size",
        type=int,
        default=256,
    )

    # --------------------------------------------------------
    # Adaptive hard-negative ranking loss
    # --------------------------------------------------------

    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--margin_scale", type=float, default=0.2)
    parser.add_argument("--hard_k", type=int, default=2)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--log_steps",
        type=int,
        default=100,
    )

    # --------------------------------------------------------
    # Misc
    # --------------------------------------------------------

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    if args.do_train:
        if args.train_data_file is None:
            raise ValueError(
                "--train_data_file is required for --do_train"
            )
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

    logging.basicConfig(
        format=(
            "%(asctime)s - %(levelname)s - "
            "%(name)s - %(message)s"
        ),
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(
        "Arguments:\n%s",
        json.dumps(
            vars(args),
            indent=4,
        ),
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    args.device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    args.n_gpu = torch.cuda.device_count()

    logger.info(
        "device: %s, n_gpu: %d",
        args.device,
        args.n_gpu,
    )

    set_seed(
        args.seed
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = RobertaTokenizer.from_pretrained(
        args.model_name_or_path
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    (
        dual_model,
        cross_encoder,
        mlp,
    ) = prepare_models(
        args,
        tokenizer,
    )

    # --------------------------------------------------------
    # Training
    #
    # IMPORTANT:
    # train file is used both as:
    #   query dataset
    #   code candidate pool
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
    # Load trained MLP after training
    # --------------------------------------------------------

    if args.do_train:
        args.mlp_checkpoint = best_mlp_checkpoint(args)
    elif (
        args.mlp_checkpoint is None
        and (args.do_eval or args.do_test)
    ):
        args.mlp_checkpoint = best_mlp_checkpoint(args)

    if args.do_train or args.do_eval or args.do_test:
        load_mlp_checkpoint(
            mlp,
            args.mlp_checkpoint,
            args.device,
        )

    # --------------------------------------------------------
    # Evaluation
    #
    # query = eval_data_file
    # candidates = codebase_file
    # --------------------------------------------------------

    if args.do_eval:

        if args.eval_data_file is None:
            raise ValueError(
                "--eval_data_file is required for --do_eval"
            )

        if args.codebase_file is None:
            raise ValueError(
                "--codebase_file is required for --do_eval"
            )

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

        logger.info(
            "***** Eval results *****"
        )

        assert result is not None
        for key, value in result.items():
            logger.info(
                "  %s = %.6f",
                key,
                value,
            )

    # --------------------------------------------------------
    # Test
    #
    # query = test_data_file
    # candidates = codebase_file
    # --------------------------------------------------------

    if args.do_test:

        if args.test_data_file is None:
            raise ValueError(
                "--test_data_file is required for --do_test"
            )

        if args.codebase_file is None:
            raise ValueError(
                "--codebase_file is required for --do_test"
            )

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

        logger.info(
            "***** Test results *****"
        )

        assert result is not None
        for key, value in result.items():
            logger.info(
                "  %s = %.6f",
                key,
                value,
            )


if __name__ == "__main__":
    main()