import os
import json
import argparse
import multiprocessing as mp
import logging
import sys
import re

import torch

from model import LanguageModel


SUMMARY_PROMPT = """Below is a {language} code that describes a task. Please give a short summary describing the purpose of the code. You must write only summary without any prefix or suffix explanations.
{code}
"""

# ============================================================
# Dataset
# ============================================================

def load_dataset(data_file):
    """
    Load JSON or JSONL dataset.

    The original record is preserved and the generated summary is added to it.
    """
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
# Helpers
# ============================================================

def split_ranges(n, k):
    """
    Split n samples into k approximately equal chunks.

    Example:
        n = 100, k = 4
        -> [(0,25), (25,50), (50,75), (75,100)]
    """
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
    """
    Parse partition string x/y.

    Example:
        1/4 -> (0, 4)
        2/4 -> (1, 4)
    """
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
    """
    Get the global [start, end) range for a partition.
    """
    chunk = total // part_total

    start = part_idx * chunk

    if part_idx == part_total - 1:
        end = total
    else:
        end = (part_idx + 1) * chunk

    return start, end


# ============================================================
# Worker
# ============================================================

def worker(
    gpu_id,
    model_path,
    data_file,
    output_file,
    batch_size,
    max_input_tokens,
    max_new_tokens,
    start,
    end,
):
    """
    One worker = one GPU = one LanguageModel.

    The worker loads the whole dataset on CPU but only processes
    [start, end).
    """

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | GPU %(process)d | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger(__name__)

    logger.info(
        f"[GPU {gpu_id}] Start | range={start}-{end}"
    )

    # --------------------------------------------------------
    # CUDA
    # --------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Multi-GPU augmentation requires CUDA."
        )

    torch.cuda.set_device(gpu_id)

    device = f"cuda:{gpu_id}"

    logger.info(
        f"[GPU {gpu_id}] Loading model on {device}"
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    data = load_dataset(data_file)

    subset = data[start:end]

    logger.info(
        f"[GPU {gpu_id}] Number of samples: {len(subset)}"
    )

    # --------------------------------------------------------
    # Load LLM
    # --------------------------------------------------------

    model = LanguageModel(
        model_path,
        device,
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    results = []

    for batch_start in range(0, len(subset), batch_size):

        batch = subset[
            batch_start:batch_start + batch_size
        ]

        logger.info(
            f"[GPU {gpu_id}] "
            f"Processing {start + batch_start}"
            f"-{start + batch_start + len(batch)}"
            f"/{end}"
        )

        # ----------------------------------------------------
        # Build prompts
        # ----------------------------------------------------

        summary_prompts = [
            SUMMARY_PROMPT.format(language=js["language"], code=js["code"])
            for js in batch
        ]

        # ----------------------------------------------------
        # Generate summaries
        # ----------------------------------------------------

        generated_summaries = model.generate(
            summary_prompts,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        for js, generated_summary in zip(batch, generated_summaries):
            result = dict(js)
            result["summary"] = generated_summary.strip()
            results.append(result)

    # --------------------------------------------------------
    # Save worker result
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(output_file) or ".",
        exist_ok=True,
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        for result in results:
            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    logger.info(
        f"[GPU {gpu_id}] Done | saved={output_file}"
    )


# ============================================================
# Run generation
# ============================================================

def run_generation(
    model_path,
    data_file,
    output_file,
    gpu_ids,
    batch_size,
    max_input_tokens,
    max_new_tokens,
    part,
    logger,
):
    """
    Multi-GPU generation pipeline.

    Dataset
        |
        +---- GPU 0 -> part file
        |
        +---- GPU 1 -> part file
        |
        +---- GPU 2 -> part file
        |
        ...
        |
        +---- merge
              |
              v
          final JSONL
    """

    logger.info("========================================")
    logger.info("Starting augmentation generation")
    logger.info("========================================")

    data = load_dataset(data_file)

    total = len(data)

    logger.info(
        f"Total samples: {total}"
    )

    # --------------------------------------------------------
    # Partition
    # --------------------------------------------------------

    part_idx, part_total = parse_part(part)

    if part_idx is not None:

        global_start, global_end = get_part_range(
            total,
            part_idx,
            part_total,
        )

        logger.info(
            f"Using partition {part_idx + 1}/{part_total}: "
            f"{global_start}-{global_end}"
        )

    else:

        global_start = 0
        global_end = total

    # --------------------------------------------------------
    # GPU ranges
    # --------------------------------------------------------

    sub_total = global_end - global_start

    ranges = split_ranges(
        sub_total,
        len(gpu_ids),
    )

    processes = []
    part_files = []

    suffix = (
        f"_part{part_idx + 1}of{part_total}"
        if part_idx is not None
        else "_full"
    )

    # --------------------------------------------------------
    # Spawn workers
    # --------------------------------------------------------

    for i, gpu_id in enumerate(gpu_ids):

        sub_start, sub_end = ranges[i]

        start = global_start + sub_start
        end = global_start + sub_end

        # Skip empty ranges
        if start >= end:
            continue

        part_file = output_file.replace(
            ".json",
            f"{suffix}_gpu{gpu_id}.json",
        )

        part_files.append(part_file)

        logger.info(
            f"Launching GPU {gpu_id}: "
            f"{start}-{end}"
        )

        p = mp.Process(
            target=worker,
            args=(
                gpu_id,
                model_path,
                data_file,
                part_file,
                batch_size,
                max_input_tokens,
                max_new_tokens,
                start,
                end,
            ),
        )

        p.start()

        processes.append(p)

    # --------------------------------------------------------
    # Wait
    # --------------------------------------------------------

    for p in processes:
        p.join()

    # --------------------------------------------------------
    # Check workers
    # --------------------------------------------------------

    failed = [
        p.pid
        for p in processes
        if p.exitcode != 0
    ]

    if failed:
        raise RuntimeError(
            f"Generation failed. Worker PIDs: {failed}"
        )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    logger.info("Merging worker outputs...")

    final = []

    for part_file in part_files:

        if not os.path.exists(part_file):

            logger.warning(
                f"Missing worker output: {part_file}"
            )

            continue

        with open(
            part_file,
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:
                line = line.strip()

                if not line:
                    continue

                final.append(
                    json.loads(line)
                )

    #
    # Since JSON serialization creates new dict objects,
    # object identity cannot be used after loading.
    #
    # Prefer URL / retrieval_idx as stable identifiers.
    #

    if all("url" in x for x in final):

        position_map = {
            str(js["url"]): i
            for i, js in enumerate(data)
            if "url" in js
        }

        final.sort(
            key=lambda x: position_map.get(
                str(x["url"]),
                len(data),
            )
        )

    elif all("retrieval_idx" in x for x in final):

        position_map = {
            str(js["retrieval_idx"]): i
            for i, js in enumerate(data)
            if "retrieval_idx" in js
        }

        final.sort(
            key=lambda x: position_map.get(
                str(x["retrieval_idx"]),
                len(data),
            )
        )

    # --------------------------------------------------------
    # Output filename
    # --------------------------------------------------------

    final_output = output_file

    if part_idx is not None:

        final_output = output_file.replace(
            ".jsonl",
            f"_part{part_idx + 1}of{part_total}.jsonl",
        )

    os.makedirs(
        os.path.dirname(final_output) or ".",
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save final JSONL
    # --------------------------------------------------------

    with open(
        final_output,
        "w",
        encoding="utf-8",
    ) as f:

        for result in final:
            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    logger.info(
        f"Saved final output: {final_output}"
    )

    logger.info(
        f"Generated records: {len(final)}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # CUDA multiprocessing
    # --------------------------------------------------------

    mp.set_start_method(
        "spawn",
        force=True,
    )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(process)d | "
            "%(levelname)s | %(message)s"
        ),
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
    )

    logger = logging.getLogger(__name__)

    # --------------------------------------------------------
    # Arguments
    #
    # KEEP THE SAME ARGS AS THE OLD VERSION
    # --------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=SUMMARY_PROMPT,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--part",
        type=str,
        default=None,
        help="Dataset partition, e.g. 1/4",
    )

    args = parser.parse_args()

    SUMMARY_PROMPT = args.prompt

    # --------------------------------------------------------
    # GPU detection
    # --------------------------------------------------------

    n_gpu = torch.cuda.device_count()

    if n_gpu == 0:

        raise RuntimeError(
            "No CUDA GPU detected."
        )

    gpu_ids = list(range(n_gpu))

    logger.info(
        f"Detected {n_gpu} GPU(s): {gpu_ids}"
    )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    run_generation(
        model_path=args.model_path,
        data_file=args.input,
        output_file=args.output,
        gpu_ids=gpu_ids,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        part=args.part,
        logger=logger,
    )
