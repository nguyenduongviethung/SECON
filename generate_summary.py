import os
import json
import argparse
import multiprocessing as mp
import logging
import sys
import re
import gc

import torch
import torch.nn.functional as F

from model import LanguageModel
from transformers import RobertaModel, RobertaTokenizer

# Dùng đúng Model/encode_text của run_encoder để embedding khớp lúc train.
from run_encoder import Model, encode_text


# ============================================================
# Prompts
# ============================================================

SUMMARY_PROMPT = """Below is a {language} code that describes a task. Please give a short summary describing the purpose of the code. You must write only summary without any prefix or suffix explanations.
{code}
"""

SUMMARY_PROMPT_NEIGHBORS = """You are summarizing a function for code search.

Your summary must be specific enough to DISTINGUISH this function from the {n_neighbors} most similar functions shown below. Do not describe behavior shared with them; describe only what makes this function unique.

First identify its exact function name and signature, then summarize only its contract: main responsibility, key inputs/outputs, side effects, fallback/edge cases, and what makes it different from the similar functions below. Preserve exact API names, parameter names, constants, file extensions, and target paths. If the function delegates, state: "calls X; does not itself do Y". Do not output code fragments, docstring fragments, or generic verbs alone. Output 2-3 concise sentences.

=== Function to summarize (language: {language}) ===
{code}

=== Similar functions (do NOT copy their behavior) ===
{neighbors}
"""


# ============================================================
# Dataset
# ============================================================

def load_dataset(data_file):
    data = []
    with open(data_file, "r", encoding="utf-8") as f:
        if data_file.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        else:
            data = json.load(f)
    return data


# ============================================================
# Encoder-based kNN (dùng Model của run_encoder)
# ============================================================

def _encode_codes_batch(
    codes,
    tokenizer,
    model,
    max_length,
    batch_size,
    device,
):
    """Mean-pooled + L2-normalised embeddings từ run_encoder.Model."""
    embs = []
    for s in range(0, len(codes), batch_size):
        chunk = codes[s:s + batch_size]
        ids_list = [
            encode_text(c, tokenizer, max_length)[1]  # encode_text -> (tokens, ids)
            for c in chunk
        ]
        inputs = torch.tensor(ids_list, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(inputs)
        embs.append(out)
    return torch.cat(embs, dim=0)  # [N, D] on device


def compute_neighbors(
    data,
    encoder_path,
    code_length,
    topk,
    batch_size,
    device,
    logger,
):
    logger.info(
        f"Computing {topk}-NN over {len(data)} codes "
        f"using encoder: {encoder_path}"
    )

    tokenizer = RobertaTokenizer.from_pretrained(encoder_path)
    encoder = RobertaModel.from_pretrained(encoder_path)

    # Bọc raw encoder bằng run_encoder.Model để forward pass
    # (mean-pool + L2-norm) khớp chính xác với lúc train.
    model_args = argparse.Namespace(frozen_layers=[])
    model = Model(encoder, model_args)
    model.to(device)
    model.eval()

    codes = [js["code"] for js in data]
    embs = _encode_codes_batch(
        codes,
        tokenizer,
        model,
        code_length,
        batch_size,
        device,
    )  # [N, D]

    N = embs.size(0)
    k = min(topk, max(N - 1, 0))

    neighbor_indices = []
    sim_batch = batch_size

    for s in range(0, N, sim_batch):
        e = min(s + sim_batch, N)
        sim = embs[s:e] @ embs.T  # [B, N]

        # Loại self
        for i in range(e - s):
            sim[i, s + i] = -1e9

        _, idx = sim.topk(k, dim=1)
        neighbor_indices.extend(idx.cpu().tolist())

    del model, encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Done computing neighbors for {N} codes.")
    return neighbor_indices


# ============================================================
# Helpers
# ============================================================

def split_ranges(n, k):
    if k <= 0:
        raise ValueError("k must be > 0")
    k = min(k, max(n, 1))
    base = n // k
    remainder = n % k
    ranges = []
    start = 0
    for i in range(k):
        size = base + (1 if i < remainder else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


def parse_part(part_str):
    if part_str is None:
        return None, None
    x, y = part_str.split("/")
    x = int(x)
    y = int(y)
    if not (1 <= x <= y):
        raise ValueError(
            f"Invalid --part={part_str}. Expected format x/y with 1 <= x <= y."
        )
    return x - 1, y


def get_part_range(total, part_idx, part_total):
    chunk = total // part_total
    start = part_idx * chunk
    if part_idx == part_total - 1:
        end = total
    else:
        end = (part_idx + 1) * chunk
    return start, end


def build_neighbors_block(nbr_codes):
    """Render block [N1]..[Nk] cho SUMMARY_PROMPT_NEIGHBORS."""
    return "\n\n".join(
        f"[N{j + 1}]\n{c}" for j, c in enumerate(nbr_codes)
    )


# ============================================================
# Worker
# ============================================================

def worker(
    gpu_id,
    model_path,
    encoder_neighbors,
    data_file,
    output_file,
    batch_size,
    max_input_tokens,
    max_new_tokens,
    summary_prompt,
    num_neighbors,
    start,
    end,
):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | GPU %(process)d | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"[GPU {gpu_id}] Start | range={start}-{end}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Multi-GPU augmentation requires CUDA."
        )

    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    logger.info(f"[GPU {gpu_id}] Loading model on {device}")

    data = load_dataset(data_file)
    subset = data[start:end]
    logger.info(f"[GPU {gpu_id}] Number of samples: {len(subset)}")

    model = LanguageModel(model_path, device)

    results = []

    for batch_start in range(0, len(subset), batch_size):
        batch = subset[batch_start:batch_start + batch_size]

        logger.info(
            f"[GPU {gpu_id}] "
            f"Processing {start + batch_start}"
            f"-{start + batch_start + len(batch)}"
            f"/{end}"
        )

        summary_prompts = []
        for i, js in enumerate(batch):
            global_idx = start + batch_start + i

            if encoder_neighbors is not None:
                nbr_idxs = encoder_neighbors[global_idx][:num_neighbors]
                nbr_codes = [data[nidx]["code"] for nidx in nbr_idxs]
            else:
                nbr_codes = []

            summary_prompts.append(
                summary_prompt.format(
                    language=js["language"],
                    code=js["code"],
                    n_neighbors=len(nbr_codes),
                    neighbors=build_neighbors_block(nbr_codes),
                )
            )

        generated_summaries = model.generate(
            summary_prompts,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
        )

        for js, generated_summary in zip(batch, generated_summaries):
            result = dict(js)
            result["summary"] = generated_summary.strip()
            results.append(result)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    logger.info(f"[GPU {gpu_id}] Done | saved={output_file}")


# ============================================================
# Run generation
# ============================================================

def run_generation(
    model_path,
    encoder_path,
    code_length,
    data_file,
    output_file,
    gpu_ids,
    batch_size,
    max_input_tokens,
    max_new_tokens,
    num_neighbors,
    part,
    summary_prompt,
    logger,
):
    logger.info("========================================")
    logger.info("Starting augmentation generation")
    logger.info("========================================")

    data = load_dataset(data_file)
    total = len(data)
    logger.info(f"Total samples: {total}")

    part_idx, part_total = parse_part(part)
    if part_idx is not None:
        global_start, global_end = get_part_range(total, part_idx, part_total)
        logger.info(
            f"Using partition {part_idx + 1}/{part_total}: "
            f"{global_start}-{global_end}"
        )
    else:
        global_start = 0
        global_end = total

    # Tính kNN MỘT LẦN ở parent process (dùng encoder)
    encoder_neighbors = None
    if encoder_path is not None and num_neighbors > 0:
        neighbor_device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        encoder_neighbors = compute_neighbors(
            data=data,
            encoder_path=encoder_path,
            code_length=code_length,
            topk=num_neighbors,
            batch_size=64,
            device=neighbor_device,
            logger=logger,
        )

    sub_total = global_end - global_start
    ranges = split_ranges(sub_total, len(gpu_ids))

    processes = []
    part_files = []

    suffix = (
        f"_part{part_idx + 1}of{part_total}"
        if part_idx is not None
        else "_full"
    )

    for i, gpu_id in enumerate(gpu_ids):
        sub_start, sub_end = ranges[i]
        start = global_start + sub_start
        end = global_start + sub_end

        if start >= end:
            continue

        part_file = output_file.replace(
            ".json",
            f"{suffix}_gpu{gpu_id}.json",
        )
        part_files.append(part_file)

        logger.info(f"Launching GPU {gpu_id}: {start}-{end}")

        p = mp.Process(
            target=worker,
            args=(
                gpu_id,
                model_path,
                encoder_neighbors,
                data_file,
                part_file,
                batch_size,
                max_input_tokens,
                max_new_tokens,
                summary_prompt,
                num_neighbors,
                start,
                end,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    failed = [p.pid for p in processes if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"Generation failed. Worker PIDs: {failed}")

    logger.info("Merging worker outputs...")

    final = []
    for part_file in part_files:
        if not os.path.exists(part_file):
            logger.warning(f"Missing worker output: {part_file}")
            continue
        with open(part_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                final.append(json.loads(line))

    if all("url" in x for x in final):
        position_map = {
            str(js["url"]): i
            for i, js in enumerate(data)
            if "url" in js
        }
        final.sort(
            key=lambda x: position_map.get(str(x["url"]), len(data))
        )
    elif all("retrieval_idx" in x for x in final):
        position_map = {
            str(js["retrieval_idx"]): i
            for i, js in enumerate(data)
            if "retrieval_idx" in js
        }
        final.sort(
            key=lambda x: position_map.get(str(x["retrieval_idx"]), len(data))
        )

    final_output = output_file
    if part_idx is not None:
        final_output = output_file.replace(
            ".jsonl",
            f"_part{part_idx + 1}of{part_total}.jsonl",
        )

    os.makedirs(os.path.dirname(final_output) or ".", exist_ok=True)

    with open(final_output, "w", encoding="utf-8") as f:
        for result in final:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    logger.info(f"Saved final output: {final_output}")
    logger.info(f"Generated records: {len(final)}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(process)d | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str)

    parser.add_argument(
        "--encoder_path",
        type=str,
        default=None,
        help="Path / model name của RoBERTa encoder đã train "
             "(vd: output_dir/checkpoint-best-mrr). Nếu có, "
             "các code gần nhất (xem --num_neighbors) sẽ được chèn vào prompt.",
    )
    parser.add_argument(
        "--code_length",
        type=int,
        default=256,
        help="Độ dài token code tối đa khi encode để tính kNN.",
    )
    parser.add_argument(
        "--num_neighbors",
        type=int,
        default=2,
        help="Số code gần nhất (theo encoder) đưa vào prompt đối chiếu. "
             "Đặt 0 để tắt neighbors. Mặc định: 2.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt. Mặc định là SUMMARY_PROMPT_NEIGHBORS khi "
             "--encoder_path được set và --num_neighbors > 0, ngược lại "
             "dùng SUMMARY_PROMPT. Prompt neighbor hỗ trợ placeholder "
             "{language}, {code}, {n_neighbors}, {neighbors}.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_input_tokens", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--part",
        type=str,
        default=None,
        help="Dataset partition, e.g. 1/4",
    )

    args = parser.parse_args()

    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        raise RuntimeError("No CUDA GPU detected.")
    gpu_ids = list(range(n_gpu))
    logger.info(f"Detected {n_gpu} GPU(s): {gpu_ids}")

    use_neighbors = (
        args.encoder_path is not None and args.num_neighbors > 0
    )

    if args.prompt is not None:
        prompt = args.prompt
    elif use_neighbors:
        prompt = SUMMARY_PROMPT_NEIGHBORS
    else:
        prompt = SUMMARY_PROMPT

    run_generation(
        model_path=args.model_path,
        encoder_path=args.encoder_path if use_neighbors else None,
        code_length=args.code_length,
        data_file=args.input,
        output_file=args.output,
        gpu_ids=gpu_ids,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        num_neighbors=args.num_neighbors,
        part=args.part,
        summary_prompt=prompt,
        logger=logger,
    )