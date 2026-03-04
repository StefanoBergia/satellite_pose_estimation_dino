"""Generate incremental self-training chunk splits from splits_testfull style files.

Reads data/splits_testfull/{domain}_style.txt for each domain,
shuffles with a fixed seed, partitions into N equal chunks, and writes
data/splits_incremental/{domain}_chunk{i}.txt.

Usage:
    python scripts/make_incremental_splits.py --n_chunks 5 --seed 42
"""

import argparse
import random
from pathlib import Path


DOMAINS = ["lightbox", "sunlamp"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_chunks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source_dir", type=str, default="data/splits_testfull")
    parser.add_argument("--out_dir", type=str, default="data/splits_incremental")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    for domain in DOMAINS:
        src = Path(args.source_dir) / f"{domain}_style.txt"
        with open(src) as f:
            lines = [l.strip() for l in f if l.strip()]

        rng_domain = random.Random(args.seed)  # same seed per domain for reproducibility
        rng_domain.shuffle(lines)

        n = len(lines)
        chunk_size = n // args.n_chunks
        remainder = n % args.n_chunks

        chunks = []
        start = 0
        for i in range(args.n_chunks):
            # distribute remainder across first chunks
            end = start + chunk_size + (1 if i < remainder else 0)
            chunks.append(lines[start:end])
            start = end

        for i, chunk in enumerate(chunks):
            out_path = out_dir / f"{domain}_chunk{i}.txt"
            with open(out_path, "w") as f:
                f.write("\n".join(chunk) + "\n")
            print(f"  {out_path}: {len(chunk)} images")

    print(f"\nDone. {args.n_chunks} chunks written to {out_dir}/")


if __name__ == "__main__":
    main()
