# coding=utf-8
"""
SECON ranker: 3-encoder retrieval with combined cosine score.

Score = semantic_weight * cos(query, semantic_code)
      + functional_weight * cos(query, functional_code)

- query_model      : SentenceTransformer, encodes natural language
- semantic_model   : SentenceTransformer, encodes code (semantic view)
- functional_model : SentenceTransformer on GraphCodeBERT, encodes code (functional view)
- Accelerate uses one process per GPU and DDP to synchronize encoder gradients.
- For training we tokenize directly and call forward (encode uses torch.no_grad).
- Only InfoNCE and adaptive hard-negative losses are kept.
- GraphCodeBERT parser is imported from repo root (no path argument needed).
"""

import argparse
import csv
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.models import Transformer, Pooling

# ------------------------------------------------------------------
# GraphCodeBERT parser utilities (assumed to be present in repo root)
# ------------------------------------------------------------------
try:
    from parser import (
        DFG_python,
        DFG_java,
        DFG_ruby,
        DFG_go,
        DFG_php,
        DFG_javascript,
        DFG_csharp,
        remove_comments_and_docstrings,
        tree_to_token_index,
        index_to_code_token,
        tree_to_variable_index,
    )
    PARSER_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).warning(
        "parser module not found (%s). Parser-based preprocessing disabled.", e
    )
    PARSER_AVAILABLE = False
    DFG_python = DFG_java = DFG_ruby = DFG_go = None
    DFG_php = DFG_javascript = DFG_csharp = None
    remove_comments_and_docstrings = None
    tree_to_token_index = index_to_code_token = tree_to_variable_index = None


logger = logging.getLogger(__name__)

DFG_FUNCTION = {
    "python": DFG_python,
    "java": DFG_java,
    "ruby": DFG_ruby,
    "go": DFG_go,
    "php": DFG_php,
    "javascript": DFG_javascript,
    "c_sharp": DFG_csharp,
}


# ============================================================
# Loss functions
# ============================================================

def infonce_from_sim(sim, temperature=0.05):
    """Symmetric InfoNCE given a similarity matrix."""
    sim = sim / temperature
    labels = torch.arange(sim.size(0), device=sim.device)
    loss_i = F.cross_entropy(sim, labels)
    loss_j = F.cross_entropy(sim.t(), labels)
    return (loss_i + loss_j) / 2


def adaptive_hard_from_sim(sim, temperature=0.05, margin_scale=0.2, hard_k=2):
    """Adaptive hard-negative ranking loss on a similarity matrix."""
    batch_size = sim.size(0)
    if batch_size <= 1:
        return torch.zeros([], device=sim.device, dtype=sim.dtype)

    pos = sim.diag()
    eye = torch.eye(batch_size, dtype=torch.bool, device=sim.device)
    neg = sim.masked_fill(eye, -1e9)

    margin = margin_scale * (1.0 - pos)
    k = min(hard_k, batch_size - 1)
    hard_neg, _ = torch.topk(neg, k=k, dim=1)
    weights = F.softmax(hard_neg / temperature, dim=1)

    ranking_loss = F.softplus(
        (hard_neg - pos.unsqueeze(1) + margin.unsqueeze(1)) / temperature
    )
    loss = (weights * ranking_loss).sum(dim=1)
    return loss.mean()


def ranker_loss(q, s, f, args):
    """Compute loss on the combined similarity matrix.

    q, s, f : [B, D] unit-normalized embeddings.
    """
    sim_sem = q @ s.t()
    sim_func = q @ f.t()
    sim = args.semantic_weight * sim_sem + args.functional_weight * sim_func

    if args.loss_type == "infonce":
        return infonce_from_sim(sim, args.temperature)
    elif args.loss_type == "adaptive_hard":
        loss1 = adaptive_hard_from_sim(
            sim,
            temperature=args.temperature,
            margin_scale=args.margin_scale,
            hard_k=args.hard_k,
        )
        loss2 = adaptive_hard_from_sim(
            sim.t(),
            temperature=args.temperature,
            margin_scale=args.margin_scale,
            hard_k=args.hard_k,
        )
        return (loss1 + loss2) / 2
    else:
        raise ValueError(f"Unknown loss_type: {args.loss_type}")


# ============================================================
# Dataset (stores raw text, no tokenization)
# ============================================================

class TextDataset(Dataset):
    def __init__(self, file_path, use_parser_preprocessing=False, language="python"):
        self.examples = []

        with open(file_path, encoding="utf-8") as f:
            if "jsonl" in file_path:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self.examples.append(json.loads(line))
            elif "json" in file_path:
                self.examples = json.load(f)

        if use_parser_preprocessing and PARSER_AVAILABLE and remove_comments_and_docstrings is not None:
            for ex in self.examples:
                code = ex.get("code")
                if not code:
                    continue
                try:
                    ex["code"], _ = remove_comments_and_docstrings(code, language)
                except Exception:
                    pass

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        return {
            "code": ex.get("code", ""),
            "nl": ex.get("docstring", ""),
            "url": ex.get("url", ex.get("retrieval_idx", str(i))),
        }


# ============================================================
# Model loading / encoding helpers
# ============================================================

def load_embedding_model(
    model_name_or_path, max_seq_length, trust_remote_code, device=None
):
    """Load a SentenceTransformer, falling back to Transformer + MeanPooling."""
    try:
        st = SentenceTransformer(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            device=device,
        )
    except Exception as e:
        logger.info(
            "Loading '%s' as full SentenceTransformer failed (%s). "
            "Building from Transformer + MeanPooling.",
            model_name_or_path, e,
        )
        transformer = Transformer(
            model_name_or_path,
            max_seq_length=max_seq_length,
            trust_remote_code=trust_remote_code,
        )
        dim = transformer.get_word_embedding_dimension()
        pooling = Pooling(dim, pooling_mode_mean_tokens=True)
        st = SentenceTransformer(modules=[transformer, pooling], device=device)

    st.max_seq_length = max_seq_length
    return st


def encode_text(model, texts, device, batch_size=32, show_progress=False):
    """Encode a list of texts to L2-normalized numpy array. No gradients."""
    model.eval()
    try:
        emb = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            device=device,
        )
    except TypeError:
        emb = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            device=device,
        )
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    return emb


def encode_for_training(model, texts, device, accelerator):
    """Encode texts with gradients flowing through `model`.

    SentenceTransformer.encode() runs under torch.no_grad(), so we call
    tokenize() + forward() directly.  The forward pass goes through the
    same modules (Transformer, Pooling, ...) as encode().
    """
    first = accelerator.unwrap_model(model)[0]
    features = first.tokenize(texts)
    features = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in features.items()
    }
    emb = model(features)
    if isinstance(emb, dict):
        emb = emb.get("sentence_embedding", list(emb.values())[-1])
    return F.normalize(emb.float(), p=2, dim=1)


# ============================================================
# Saving / loading checkpoints
# ============================================================

def best_checkpoint_dir(args):
    return os.path.join(args.output_dir, "checkpoint-best-mrr")


def save_best_models(args, query_model, semantic_model, functional_model):
    base = best_checkpoint_dir(args)
    os.makedirs(base, exist_ok=True)
    # SentenceTransformer.save() internally calls save_pretrained on the
    # underlying transformer and persists pooling / ST config files.
    query_model.save(os.path.join(base, "query_model"))
    semantic_model.save(os.path.join(base, "semantic_model"))
    functional_model.save(os.path.join(base, "functional_model"))
    logger.info("Saved best-MRR models to %s", base)


def load_best_models(args):
    base = best_checkpoint_dir(args)
    q = SentenceTransformer(
        os.path.join(base, "query_model"), device=args.device
    )
    s = SentenceTransformer(
        os.path.join(base, "semantic_model"), device=args.device
    )
    f = SentenceTransformer(
        os.path.join(base, "functional_model"), device=args.device
    )

    q.max_seq_length = args.nl_length
    s.max_seq_length = args.code_length
    f.max_seq_length = args.code_length

    q.to(args.device)
    s.to(args.device)
    f.to(args.device)

    logger.info("Loaded best-MRR models from %s", base)
    return q, s, f


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


def train(args, query_model, semantic_model, functional_model, accelerator):
    train_dataset = TextDataset(
        args.train_data_file,
        use_parser_preprocessing=args.use_parser_preprocessing,
        language=args.code_language,
    )
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )

    params = (
        list(query_model.parameters())
        + list(semantic_model.parameters())
        + list(functional_model.parameters())
    )
    optimizer = AdamW(params, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * args.num_train_epochs,
    )
    (
        query_model,
        semantic_model,
        functional_model,
        optimizer,
        train_dataloader,
        scheduler,
    ) = accelerator.prepare(
        query_model,
        semantic_model,
        functional_model,
        optimizer,
        train_dataloader,
        scheduler,
    )
    params = (
        list(query_model.parameters())
        + list(semantic_model.parameters())
        + list(functional_model.parameters())
    )

    if accelerator.is_main_process:
        logger.info("***** Running ranker training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Batch size per process = %d", args.train_batch_size)
        logger.info(
            "  Total steps per process = %d",
            len(train_dataloader) * args.num_train_epochs,
        )

    best_mrr = float("-inf")

    for epoch in range(args.num_train_epochs):
        query_model.train()
        semantic_model.train()
        functional_model.train()

        tr_loss, tr_num = 0.0, 0

        for step, batch in enumerate(train_dataloader):
            code_texts = batch["code"]
            nl_texts = batch["nl"]

            with accelerator.autocast():
                q = encode_for_training(
                    query_model, nl_texts, args.device, accelerator
                )
                s = encode_for_training(
                    semantic_model, code_texts, args.device, accelerator
                )
                f = encode_for_training(
                    functional_model, code_texts, args.device, accelerator
                )

            loss = ranker_loss(q, s, f, args)

            tr_loss += loss.item()
            tr_num += 1
            if accelerator.is_main_process and (step + 1) % 100 == 0:
                logger.info(
                    "epoch %d step %d loss %.5f",
                    epoch, step + 1, tr_loss / tr_num,
                )
                tr_loss, tr_num = 0.0, 0

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # ---- end of epoch evaluation ----
        if not args.eval_data_file:
            if accelerator.is_main_process:
                logger.warning("No --eval_data_file set; skipping per-epoch eval.")
            continue

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            results = evaluate(
                args,
                accelerator.unwrap_model(query_model),
                accelerator.unwrap_model(semantic_model),
                accelerator.unwrap_model(functional_model),
                args.eval_data_file,
            )
            for k, v in results.items():
                logger.info("  %s = %s", k, round(v, 4))

            if results["eval_mrr"] > best_mrr:
                best_mrr = results["eval_mrr"]
                logger.info("********************")
                logger.info("New best MRR: %s", round(best_mrr, 4))
                logger.info("********************")
                save_best_models(
                    args,
                    accelerator.unwrap_model(query_model),
                    accelerator.unwrap_model(semantic_model),
                    accelerator.unwrap_model(functional_model),
                )
        accelerator.wait_for_everyone()

    accelerator.free_memory(
        query_model,
        semantic_model,
        functional_model,
        optimizer,
        train_dataloader,
        scheduler,
    )

# ============================================================
# Evaluation / test
# ============================================================

def evaluate(args, query_model, semantic_model, functional_model,
             file_name, export_path=None):
    query_dataset = TextDataset(
        file_name,
        use_parser_preprocessing=args.use_parser_preprocessing,
        language=args.code_language,
    )
    code_dataset = TextDataset(
        args.codebase_file,
        use_parser_preprocessing=args.use_parser_preprocessing,
        language=args.code_language,
    )

    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    nl_texts = [ex.get("docstring", "") for ex in query_dataset.examples]
    code_texts = [ex.get("code", "") for ex in code_dataset.examples]

    nl_vecs = encode_text(
        query_model, nl_texts, args.device,
        batch_size=args.eval_batch_size,
    )
    code_sem_vecs = encode_text(
        semantic_model, code_texts, args.device,
        batch_size=args.eval_batch_size,
    )
    code_func_vecs = encode_text(
        functional_model, code_texts, args.device,
        batch_size=args.eval_batch_size,
    )

    sim_sem = nl_vecs @ code_sem_vecs.T
    sim_func = nl_vecs @ code_func_vecs.T
    scores = args.semantic_weight * sim_sem + args.functional_weight * sim_func

    sort_ids = np.argsort(scores, axis=-1, kind="quicksort")[:, ::-1]

    nl_urls = [
        ex.get("url", ex.get("retrieval_idx", str(i)))
        for i, ex in enumerate(query_dataset.examples)
    ]
    code_urls = [
        ex.get("url", ex.get("retrieval_idx", str(i)))
        for i, ex in enumerate(code_dataset.examples)
    ]

    ranks = []
    for url, sort_id in zip(nl_urls, sort_ids):
        rank = 0
        find = False
        for idx in sort_id[:1000]:
            if not find:
                rank += 1
            if code_urls[idx] == url:
                find = True
        ranks.append(1.0 / rank if find else 0.0)

    if export_path:
        _export_topk(
            args, tokenizer=None,
            query_dataset=query_dataset, code_dataset=code_dataset,
            nl_urls=nl_urls, code_urls=code_urls,
            sort_ids=sort_ids,
            sim_sem=sim_sem, sim_func=sim_func, scores=scores,
            export_path=export_path,
        )

    return {"eval_mrr": float(np.mean(ranks))}


def _export_topk(args, tokenizer,
                 query_dataset, code_dataset,
                 nl_urls, code_urls,
                 sort_ids, sim_sem, sim_func, scores,
                 export_path):
    rows = []
    topk = min(args.topk, len(code_dataset))
    for query_index, candidate_indices in enumerate(sort_ids[:, :topk]):
        query_example = query_dataset.examples[query_index]
        query_text = query_example.get("docstring", "")
        query_url = nl_urls[query_index]

        for rank, code_idx in enumerate(candidate_indices, start=1):
            code_idx = int(code_idx)
            code_example = code_dataset.examples[code_idx]
            rows.append({
                "query_index": query_index,
                "query_url": query_url,
                "query": query_text,
                "codebase_index": code_idx,
                "code_url": code_urls[code_idx],
                "code": code_example.get("code", ""),
                "ranker_topk_rank": rank,
                "query_semantic_score": float(sim_sem[query_index, code_idx]),
                "query_functional_score": float(sim_func[query_index, code_idx]),
                "total_score": float(scores[query_index, code_idx]),
                "is_positive": int(code_urls[code_idx] == query_url),
            })

    os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
    fields = [
        "query_index", "query_url", "query",
        "codebase_index", "code_url", "code",
        "ranker_topk_rank",
        "query_semantic_score", "query_functional_score",
        "total_score", "is_positive",
    ]
    with open(export_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Exported %d query-code pairs to %s", len(rows), export_path)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    # ---------------- Data ----------------
    parser.add_argument("--train_data_file", default=None, type=str)
    parser.add_argument("--eval_data_file", default=None, type=str)
    parser.add_argument("--test_data_file", default=None, type=str)
    parser.add_argument("--codebase_file", default=None, type=str)
    parser.add_argument("--test_results_file", default=None, type=str)
    parser.add_argument("--output_dir", required=True, type=str)

    # ---------------- Model weights ----------------
    parser.add_argument(
        "--query_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer checkpoint for the natural-language query encoder.",
    )
    parser.add_argument(
        "--semantic_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer checkpoint for the code semantic encoder.",
    )
    parser.add_argument(
        "--functional_model", type=str, default="microsoft/graphcodebert-base",
        help="SentenceTransformer / HF checkpoint for the code functional encoder.",
    )

    # ---------------- Trust remote code ----------------
    parser.add_argument("--query_trust_remote_code", action="store_true")
    parser.add_argument("--semantic_trust_remote_code", action="store_true")
    parser.add_argument("--functional_trust_remote_code", action="store_true")

    # ---------------- Scoring weights ----------------
    parser.add_argument("--semantic_weight", type=float, default=0.5)
    parser.add_argument("--functional_weight", type=float, default=0.5)

    # ---------------- Sequence lengths ----------------
    parser.add_argument("--nl_length", default=128, type=int)
    parser.add_argument("--code_length", default=256, type=int)

    # ---------------- Modes ----------------
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--do_zero_shot", action="store_true")

    # ---------------- Training ----------------
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)

    parser.add_argument(
        "--loss_type", default="infonce",
        choices=["infonce", "adaptive_hard"],
        type=str,
    )
    parser.add_argument("--temperature", default=0.05, type=float)
    parser.add_argument("--margin_scale", default=0.2, type=float)
    parser.add_argument("--hard_k", default=2, type=int)

    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=1, type=int)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--topk", default=100, type=int)
    parser.add_argument(
        "--use_parser_preprocessing", action="store_true",
        help="Apply GraphCodeBERT parser's remove_comments_and_docstrings on code.",
    )
    parser.add_argument("--code_language", default="python", type=str)

    args = parser.parse_args()

    if args.do_test and args.topk <= 0:
        parser.error("--topk must be greater than 0")
    if args.do_train and not args.eval_data_file:
        logger.warning("Training without --eval_data_file; per-epoch eval is skipped.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Base encoders may contain parameters, such as a pooler, unused by embeddings.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    print(json.dumps(vars(args), indent=4))

    device = accelerator.device
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info(
        "device: %s, n_gpu: %s, process: %s/%s",
        device,
        args.n_gpu,
        accelerator.process_index,
        accelerator.num_processes,
    )

    set_seed(args.seed)

    # ---------------- Build models ----------------
    query_model = semantic_model = functional_model = None
    if args.do_train or args.do_zero_shot:
        query_model = load_embedding_model(
            args.query_model,
            args.nl_length,
            args.query_trust_remote_code,
            device=args.device,
        )
        semantic_model = load_embedding_model(
            args.semantic_model,
            args.code_length,
            args.semantic_trust_remote_code,
            device=args.device,
        )
        functional_model = load_embedding_model(
            args.functional_model,
            args.code_length,
            args.functional_trust_remote_code,
            device=args.device,
        )

        query_model.to(args.device)
        semantic_model.to(args.device)
        functional_model.to(args.device)

    # ---------------- Train ----------------
    if args.do_train:
        train(args, query_model, semantic_model, functional_model, accelerator)

    models_loaded_from_checkpoint = False
    if args.do_train and not args.do_zero_shot and (args.do_eval or args.do_test):
        query_model = semantic_model = functional_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------- Eval ----------------
    if args.do_eval:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if not args.do_zero_shot:
                query_model, semantic_model, functional_model = load_best_models(args)
                models_loaded_from_checkpoint = True
            else:
                query_model = accelerator.unwrap_model(query_model)
                semantic_model = accelerator.unwrap_model(semantic_model)
                functional_model = accelerator.unwrap_model(functional_model)

            result = evaluate(
                args, query_model, semantic_model, functional_model,
                args.eval_data_file,
            )
            logger.info("***** Eval results *****")
            for k in sorted(result.keys()):
                logger.info("  %s = %s", k, round(result[k], 4))
        accelerator.wait_for_everyone()

    # ---------------- Test ----------------
    if args.do_test:
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            return

        if not args.do_zero_shot:
            if not models_loaded_from_checkpoint:
                query_model, semantic_model, functional_model = load_best_models(args)

        export_path = args.test_results_file or os.path.join(
            args.output_dir, "ranker_test_results.csv"
        )
        result = evaluate(
            args, query_model, semantic_model, functional_model,
            args.test_data_file,
            export_path=export_path,
        )
        logger.info("***** Test results *****")
        for k in sorted(result.keys()):
            logger.info("  %s = %s", k, round(result[k], 4))


if __name__ == "__main__":
    main()