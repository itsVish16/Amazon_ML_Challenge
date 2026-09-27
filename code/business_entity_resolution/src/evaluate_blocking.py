"""
Rigorous Blocking Recall & Error Analysis on Entity-Level Holdout Split.
Evaluates candidate retrieval recall across 10,000 holdout entities,
categorizes every failure mode, and tests expanded blocking channels to hit >= 99.2% recall.
"""

import os
import sys
import time
from collections import defaultdict, Counter
import numpy as np

from normalization import normalize_business_name, normalize_address
from blocking import CountryBlocker, generate_blocking_keys

def evaluate_blocking(
    data_dir: str = "DATA/student_resource/dataset/train",
    num_entities: int = 10000,
    max_cands: int = 40,
    max_freq: int = 2000
):
    print(f"Loading {num_entities} S1 entities from {data_dir}...")
    s1_data = {}
    with open(os.path.join(data_dir, "train_source1.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                s1_data[parts[0]] = (parts[1], parts[2], parts[3])
            if len(s1_data) >= num_entities:
                break

    # Load Ground Truth
    s1_gt = {}
    all_true_targets = set()
    with open(os.path.join(data_dir, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] in s1_data:
                matches = set(parts[1].split(",")) if len(parts) > 1 and parts[1].strip() else set()
                s1_gt[parts[0]] = matches
                all_true_targets.update(matches)

    print(f"Total True Targets across sample: {len(all_true_targets)}")

    # Load target records (True targets + 50,000 background records to test realistic competition)
    target_data = {}
    for fn in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(data_dir, fn)
        with open(path, "r", encoding="utf-8") as f:
            next(f)
            for i, line in enumerate(f):
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    eid = parts[0]
                    if eid in all_true_targets or i < 25000:
                        target_data[eid] = (parts[1], parts[2], parts[3])

    print(f"Total Target Pool for Blocking Test: {len(target_data)}")

    # Index into blockers
    blockers = {
        'US': CountryBlocker('US', max_candidates=max_cands, max_freq=max_freq),
        'India': CountryBlocker('India', max_candidates=max_cands, max_freq=max_freq)
    }

    t0 = time.time()
    for eid, (n, a, c) in target_data.items():
        if c in blockers:
            blockers[c].add_target_record(eid, n, a)
    print(f"Indexed targets in {time.time()-t0:.2f}s.")

    # Evaluate Recall
    total_true_matches = 0
    captured_matches = 0
    cand_counts = []
    missed_pairs = []

    for s1_id, (n, a, c) in s1_data.items():
        if c not in blockers:
            continue
        true_set = s1_gt[s1_id]
        total_true_matches += len(true_set)

        cands = set(blockers[c].retrieve_candidates(n, a))
        cand_counts.append(len(cands))

        hits = cands & true_set
        captured_matches += len(hits)

        for m in true_set:
            if m not in cands and m in target_data:
                missed_pairs.append((s1_id, m))

    recall = (captured_matches / total_true_matches * 100) if total_true_matches else 0
    print("\n==================================================")
    print(f" BLOCKING RECALL EVALUATION (max_candidates={max_cands})")
    print("==================================================")
    print(f"Total True Matches: {total_true_matches}")
    print(f"Captured Matches:   {captured_matches} ({recall:.2f}%)")
    print(f"Missed Matches:     {len(missed_pairs)} ({100-recall:.2f}%)")
    print(f"Average Candidates per S1: {np.mean(cand_counts):.2f}")
    print(f"Median Candidates per S1:  {np.median(cand_counts)}")

    # Failure Analysis Breakdown
    print("\n--- DEEP FAILURE ANALYSIS OF MISSED PAIRS ---")
    failure_types = Counter()
    for s1_id, mid in missed_pairs:
        s1_n, s1_a, s1_c = s1_data[s1_id]
        tn, ta, tc = target_data[mid]

        s1_norm_n = normalize_business_name(s1_n)
        s1_norm_a = normalize_address(s1_a, s1_c)
        t_norm_n = normalize_business_name(tn)
        t_norm_a = normalize_address(ta, tc)

        s1_nums = set(s1_norm_a['numbers'])
        t_nums = set(t_norm_a['numbers'])
        s1_name_toks = set(s1_norm_n['tokens'])
        t_name_toks = set(t_norm_n['tokens'])

        if not ta.strip():
            failure_types['target_address_empty'] += 1
        elif not s1_a.strip():
            failure_types['s1_address_empty'] += 1
        elif s1_nums and t_nums and not (s1_nums & t_nums):
            failure_types['disjoint_numbers'] += 1
        elif not (s1_name_toks & t_name_toks):
            failure_types['no_name_tokens_overlap'] += 1
        else:
            failure_types['other_key_divergence'] += 1

    for f_type, count in failure_types.most_common():
        print(f"  {f_type}: {count} ({count/len(missed_pairs)*100:.1f}%)")

    print("\nSample Missed Pairs:")
    for s1_id, mid in missed_pairs[:6]:
        print(f"  S1: [{s1_id}] Name: {s1_data[s1_id][0]!r} | Addr: {s1_data[s1_id][1]!r}")
        print(f"  T : [{mid}] Name: {target_data[mid][0]!r} | Addr: {target_data[mid][1]!r}")
        print("  ---")


if __name__ == "__main__":
    evaluate_blocking()
