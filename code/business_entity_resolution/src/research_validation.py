"""
Rigorous Full-Target Validation Benchmark for Business Entity Resolution.
Evaluates:
  1. Full-Target Universe Blocking Recall (top 25, 40, 50) with vs without 250-cap
  2. Average Candidates Evaluated
  3. Candidate Truncation and Negative Distribution Shifts
  4. Final Decision Precision, Singleton Accuracy, and Macro F0.5
"""

import os
import sys
import time
import zlib
import argparse
from collections import defaultdict
import numpy as np
import lightgbm as lgb
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(__file__))
from normalization import normalize_business_name, normalize_address
from features import compute_pair_features, FEATURE_NAMES
from decision import should_accept_pair, should_skip_pair, are_states_compatible


def compute_entity_f05(true_set: set, predicted_set: set) -> float:
    if not true_set and not predicted_set:
        return 1.0
    if not true_set or not predicted_set:
        return 0.0
    true_positive = len(true_set & predicted_set)
    if not true_positive:
        return 0.0
    precision = true_positive / len(predicted_set)
    recall = true_positive / len(true_set)
    return 1.25 * precision * recall / (0.25 * precision + recall)


class FullUniverseBlocker:
    """Inverted index blocker with support for capped vs uncapped candidate accumulation."""
    def __init__(self, country: str, max_freq: int = 2500):
        self.country = country
        self.max_freq = max_freq
        self.index = defaultdict(list)
        self.key_frequencies = defaultdict(int)

    def add_target_record(self, entity_id: str, raw_name: str, raw_addr: str):
        from blocking import generate_blocking_keys
        keys = generate_blocking_keys(raw_name, raw_addr, self.country)
        for k in keys:
            self.index[k].append(entity_id)
            self.key_frequencies[k] += 1

    def retrieve_candidates(self, raw_name: str, raw_addr: str, top_k: int = 25, use_cap: bool = False) -> list:
        from blocking import generate_blocking_keys
        keys = generate_blocking_keys(raw_name, raw_addr, self.country)
        if not keys:
            return []

        scored = [(self.key_frequencies.get(k, 0), k) for k in keys if k in self.index]
        scored.sort(key=lambda x: x[0])

        if use_cap:
            # Flawed 250-cap logic (baseline)
            candidates = {}
            candidate_cap = max(top_k * 10, 250)
            for freq, k in scored:
                if freq > self.max_freq:
                    continue
                key_weight = 1.0 / (1.0 + freq)
                for cid in self.index[k]:
                    if cid in candidates:
                        candidates[cid] += key_weight
                    elif len(candidates) < candidate_cap:
                        candidates[cid] = key_weight
            return [cid for cid, _ in sorted(candidates.items(), key=lambda item: (-item[1], item[0]))[:top_k]]
        else:
            # Uncapped full candidate accumulation
            candidates = defaultdict(float)
            for freq, k in scored:
                if freq > self.max_freq:
                    continue
                key_weight = 1.0 / (1.0 + freq)
                for cid in self.index[k]:
                    candidates[cid] += key_weight
            return [cid for cid, _ in sorted(candidates.items(), key=lambda item: (-item[1], item[0]))[:top_k]]


def run_benchmark(
    data_dir: str = "DATA/student_resource/dataset/train",
    country: str = "US",
    sample_s1: int = 5000,
    seed: int = 20260927,
    model_path: str = None,
    threshold: float = 0.70,
):
    print("=================================================================")
    print(f" Research Validation Benchmark: Country={country}, Sample={sample_s1:,}")
    print("=================================================================")

    # 1. Load full target records for this country
    print(f"\n[1/4] Indexing FULL Target Universe for {country}...")
    t0 = time.time()
    blocker = FullUniverseBlocker(country, max_freq=2500)
    raw_targets = {}
    total_targets = 0

    for fn in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(data_dir, fn)
        print(f"  Reading {fn}...")
        with open(path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4 and parts[3] == country:
                    eid, name, addr = parts[0], parts[1], parts[2]
                    blocker.add_target_record(eid, name, addr)
                    raw_targets[eid] = (name, addr)
                    total_targets += 1

    print(f"  Indexed {total_targets:,} full targets in {time.time()-t0:.1f}s.")
    print(f"  Total inverted index keys: {len(blocker.index):,}")

    # 2. Load Holdout Source 1 entities with ground truth
    print(f"\n[2/4] Loading Holdout Source 1 Entities...")
    s1_holdout = []
    with open(os.path.join(data_dir, "train_source1.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and parts[3] == country:
                eid, name, addr = parts[0], parts[1], parts[2]
                split_val = zlib.crc32(eid.encode("utf-8")) & 0xFF
                # Use deterministic validation partition (crc32 >= 205, ~20% of data)
                if split_val >= 205:
                    s1_holdout.append((eid, name, addr))
                    if len(s1_holdout) >= sample_s1:
                        break

    s1_ids = {eid for eid, _, _ in s1_holdout}
    print(f"  Selected {len(s1_holdout):,} holdout S1 entities.")

    labels = {}
    with open(os.path.join(data_dir, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0] in s1_ids:
                matches = set(parts[1].split(",")) if len(parts) > 1 and parts[1] else set()
                labels[parts[0]] = matches

    singletons_count = sum(1 for m in labels.values() if not m)
    total_true_links = sum(len(m) for m in labels.values())
    print(f"  Total true links: {total_true_links:,}; Singletons: {singletons_count:,} ({singletons_count/len(s1_holdout)*100:.1f}%)")

    # 3. Benchmark Blocker: 250-Cap vs Uncapped & K=25, 40, 50
    print("\n[3/4] Benchmarking Blocking Recall Against Full Target Universe...")
    configs = [
        ("Top-25 (with 250-Cap, Current)", 25, True),
        ("Top-25 (Uncapped, Proposed)", 25, False),
        ("Top-40 (Uncapped)", 40, False),
        ("Top-50 (Uncapped)", 50, False),
    ]

    for name, top_k, use_cap in configs:
        retrieved_links = 0
        cand_counts = []
        t_block = time.time()
        for eid, s1_name, s1_addr in s1_holdout:
            cands = blocker.retrieve_candidates(s1_name, s1_addr, top_k=top_k, use_cap=use_cap)
            cand_counts.append(len(cands))
            true_set = labels[eid]
            retrieved_links += len(set(cands) & true_set)
        elapsed = time.time() - t_block
        recall = retrieved_links / max(total_true_links, 1)
        avg_cand = np.mean(cand_counts)
        rate = len(s1_holdout) / max(elapsed, 0.001)
        print(f"  {name:32s} | Recall: {recall*100:6.2f}% ({retrieved_links:,}/{total_true_links:,}) | Avg Cand: {avg_cand:4.1f} | Speed: {rate:5.0f} ent/s")

    # 4. End-to-End Scoring Benchmark with Model & Decision Rules
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "matching_model.txt")

    if not os.path.exists(model_path):
        print(f"\n[4/4] Model file {model_path} not found. Skipping model evaluation.")
        return

    print(f"\n[4/4] Evaluating End-to-End Precision & Macro F0.5 with Model...")
    model = lgb.Booster(model_file=model_path)

    # Pre-normalize S1 entities
    s1_norm = {
        eid: {
            "name": normalize_business_name(name),
            "addr": normalize_address(addr, country),
            "country": country,
        }
        for eid, name, addr in s1_holdout
    }

    # Test two candidate settings: Top-25 (uncapped) and Top-50 (uncapped)
    for test_k in [25, 50]:
        print(f"\n  --- Testing Top-{test_k} Candidates (Uncapped) with Threshold={threshold:.2f} ---")
        target_cache = {}
        all_scores = []
        singleton_scores = []
        non_singleton_scores = []
        total_pred_matches = 0
        total_true_positives = 0
        total_false_positives = 0

        t_eval = time.time()
        for eid, s1_name, s1_addr in s1_holdout:
            s_n = s1_norm[eid]
            true_set = labels[eid]
            cands = blocker.retrieve_candidates(s1_name, s1_addr, top_k=test_k, use_cap=False)

            features = []
            c_nodes = []
            for cid in cands:
                raw_t = raw_targets.get(cid)
                if raw_t is None:
                    continue
                cn = target_cache.get(cid)
                if cn is None:
                    cn = {
                        "name": normalize_business_name(raw_t[0]),
                        "addr": normalize_address(raw_t[1], country),
                    }
                    target_cache[cid] = cn

                if should_skip_pair(s_n, cn):
                    continue
                features.append(compute_pair_features(s_n, cn, cid))
                c_nodes.append((cid, cn))

            predicted = set()
            if features:
                probs = model.predict(np.asarray(features, dtype=np.float32), num_threads=1)
                for (cid, cn), prob in zip(c_nodes, probs):
                    if should_accept_pair(s_n, cn, prob, threshold):
                        predicted.add(cid)

            f05 = compute_entity_f05(true_set, predicted)
            all_scores.append(f05)
            if not true_set:
                singleton_scores.append(f05)
            else:
                non_singleton_scores.append(f05)

            tp = len(true_set & predicted)
            fp = len(predicted - true_set)
            total_true_positives += tp
            total_false_positives += fp
            total_pred_matches += len(predicted)

        macro_f05 = float(np.mean(all_scores))
        singleton_f05 = float(np.mean(singleton_scores)) if singleton_scores else 0.0
        non_singleton_f05 = float(np.mean(non_singleton_scores)) if non_singleton_scores else 0.0
        precision = total_true_positives / max(total_pred_matches, 1)
        recall = total_true_positives / max(total_true_links, 1)

        print(f"  FULL-DATA BLOCKING RECALL: {recall*100:.2f}%")
        print(f"  AVG CANDIDATES RETRIEVED: {test_k}")
        print(f"  FINAL PRECISION:          {precision*100:.2f}% ({total_true_positives:,}/{total_pred_matches:,})")
        print(f"  TOTAL FALSE POSITIVES:    {total_false_positives:,}")
        print(f"  SINGLETON F0.5 ACCURACY:  {singleton_f05:.4f}")
        print(f"  NON-SINGLETON F0.5:       {non_singleton_f05:.4f}")
        print(f"  >>> OVERALL MACRO F0.5:   {macro_f05:.5f} <<<")
        print(f"  Evaluated in {time.time()-t_eval:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="DATA/student_resource/dataset/train")
    parser.add_argument("--country", default="US")
    parser.add_argument("--sample-s1", type=int, default=3000)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    run_benchmark(
        data_dir=args.data_dir,
        country=args.country,
        sample_s1=args.sample_s1,
        threshold=args.threshold,
        model_path=args.model_path,
    )
