#!/usr/bin/env python3
"""Analyze the Australian Legal Corpus to find suitable subsets."""

import sys
from collections import defaultdict

from datasets import load_dataset


def analyze_corpus(sample_size: int = 10000):
    """Analyze corpus distribution by jurisdiction and source."""

    print("Loading dataset (streaming mode)...")
    dataset = load_dataset(
        "umarbutler/open-australian-legal-corpus",
        split="corpus",
        streaming=True,
    )

    # Counters
    by_jurisdiction = defaultdict(int)
    by_type = defaultdict(int)
    by_source = defaultdict(int)
    by_jurisdiction_type = defaultdict(int)
    total_chars = defaultdict(int)
    doc_count = 0

    print(f"Analyzing first {sample_size} documents...")

    for doc in dataset.take(sample_size):
        jurisdiction = doc.get("jurisdiction", "unknown")
        doc_type = doc.get("type", "unknown")
        source = doc.get("source", "unknown")
        text = doc.get("text", "")

        by_jurisdiction[jurisdiction] += 1
        by_type[doc_type] += 1
        by_source[source] += 1
        by_jurisdiction_type[f"{jurisdiction}:{doc_type}"] += 1
        total_chars[jurisdiction] += len(text)
        doc_count += 1

        if doc_count % 1000 == 0:
            print(f"  Processed {doc_count} documents...")

    # Scale estimates to full corpus (232,560 docs)
    scale_factor = 232560 / sample_size

    print("\n" + "=" * 70)
    print("CORPUS ANALYSIS (extrapolated from sample)")
    print("=" * 70)

    print("\n## By Jurisdiction (estimated full corpus)")
    print("-" * 50)
    for jurisdiction, count in sorted(by_jurisdiction.items(), key=lambda x: -x[1]):
        est_count = int(count * scale_factor)
        est_chars = int(total_chars[jurisdiction] * scale_factor)
        est_tokens = est_chars // 4  # rough estimate
        est_cost = est_tokens * 0.02 / 1_000_000
        print(
            f"{jurisdiction:25} {est_count:>8} docs  ~{est_tokens / 1e6:>5.1f}M tokens  ~${est_cost:>5.2f}"
        )

    print("\n## By Document Type (estimated full corpus)")
    print("-" * 50)
    for doc_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
        est_count = int(count * scale_factor)
        print(f"{doc_type:25} {est_count:>8} docs")

    print("\n## By Source (top 15)")
    print("-" * 50)
    for source, count in sorted(by_source.items(), key=lambda x: -x[1])[:15]:
        est_count = int(count * scale_factor)
        print(f"{source:40} {est_count:>8} docs")

    print("\n## Recommended Subsets (under $20)")
    print("-" * 50)

    # Find subsets under $20
    recommendations = []
    for jurisdiction, count in by_jurisdiction.items():
        est_chars = int(total_chars[jurisdiction] * scale_factor)
        est_tokens = est_chars // 4
        est_cost = est_tokens * 0.02 / 1_000_000
        est_count = int(count * scale_factor)
        if est_cost <= 20:
            recommendations.append((jurisdiction, est_count, est_tokens, est_cost))

    recommendations.sort(key=lambda x: -x[1])  # Sort by doc count

    for jurisdiction, est_count, est_tokens, est_cost in recommendations:
        print(f"{jurisdiction:25} {est_count:>8} docs  ~${est_cost:>5.2f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    analyze_corpus(sample_size)
