"""
High-Throughput Multi-Core Parallel Entity Resolution Inference Pipeline.
Optimized for 64 vCPU & High-Memory Compute Instances (Nebius / Cloud / Local M-Series).

Features:
  1. Multi-Process Worker Pool (ProcessPoolExecutor) saturating all logical CPU cores.
  2. Zero-Copy In-Memory Target & Inverted-Index Caching with OS Copy-On-Write sharing.
  3. Microsecond early-exit filtering on disjoint house numbers and name tokens.
  4. Multi-Tiered Decision Engine with strict state-conflict and street-conflict pruning.
  5. Exact row-for-row alignment matching test_source1.tsv.
  6. Automated post-run validation with validate_submission.py.
"""

import os
import sys
import time
import math
import gc
import pickle
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import numpy as np
import lightgbm as lgb

from normalization import normalize_business_name, normalize_address
from blocking import CountryBlocker
from features import compute_pair_features, FEATURE_NAMES
from decision import should_accept_pair, should_skip_pair

# Worker Process Global State
_WORKER_BLOCKER = None
_WORKER_TARGETS = None
_WORKER_MODEL = None
_WORKER_THRESHOLD = 0.70
_WORKER_COUNTRY = None


def init_worker(blocker, targets, model_path: str, threshold: float, country: str):
    """Initialize worker process with shared in-memory blocker, targets, and model."""
    global _WORKER_BLOCKER, _WORKER_TARGETS, _WORKER_MODEL, _WORKER_THRESHOLD, _WORKER_COUNTRY
    _WORKER_BLOCKER = blocker
    _WORKER_TARGETS = targets
    _WORKER_THRESHOLD = threshold
    _WORKER_COUNTRY = country
    # Initialize LightGBM booster with num_threads=1 to prevent CPU oversubscription across workers
    _WORKER_MODEL = lgb.Booster(model_file=model_path)


def process_s1_chunk(chunk: list) -> list:
    """Process a chunk of Source 1 entities across candidates and decision rules."""
    chunk_results = []

    for row_index, s1_id, s1_name, s1_addr in chunk:
        s1_n = normalize_business_name(s1_name)
        s1_a = normalize_address(s1_addr, _WORKER_COUNTRY)
        s1_norm = {
            'name': s1_n,
            'addr': s1_a,
            'country': _WORKER_COUNTRY,
        }

        cands = _WORKER_BLOCKER.retrieve_candidates(s1_name, s1_addr)

        features = []
        c_nodes = []
        for cid in cands:
            cn = _WORKER_TARGETS.get(cid)
            if cn is None or should_skip_pair(s1_norm, cn):
                continue
            features.append(compute_pair_features(s1_norm, cn, cid))
            c_nodes.append((cid, cn))

        matched = []
        if features:
            X = np.asarray(features, dtype=np.float32)
            probs = _WORKER_MODEL.predict(X, num_threads=1)
            for (cid, cn), prob in zip(c_nodes, probs):
                if should_accept_pair(s1_norm, cn, prob, _WORKER_THRESHOLD):
                    matched.append(cid)

        all_cands = sorted(set(cands) | set(matched))
        chunk_results.append((row_index, s1_id, ",".join(matched), ",".join(all_cands)))

    return chunk_results


def run_pipeline(
    test_dir: str = "DATA/student_resource/dataset/test",
    output_dir: str = "output",
    model_path: str = None,
    meta_path: str = None,
    threshold: float = 0.69,
    batch_size: int = 5000,
    max_candidates: int = 25,
    num_workers: int = None,
):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    src_dir = os.path.dirname(__file__)
    if model_path is None:
        model_path = os.path.join(src_dir, "matching_model.txt")
    if meta_path is None:
        meta_path = os.path.join(src_dir, "model_meta.pkl")

    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
            threshold = meta.get("threshold", threshold)
            expected_features = meta.get("feature_names")
            if expected_features != FEATURE_NAMES:
                raise RuntimeError(
                    "Model metadata does not match active feature contract. Retrain with src/train.py."
                )

    # Validate model integrity
    test_booster = lgb.Booster(model_file=model_path)
    if test_booster.num_feature() != len(FEATURE_NAMES) or test_booster.feature_name() != FEATURE_NAMES:
        raise RuntimeError("LightGBM model does not match active feature contract.")
    del test_booster

    # Configure worker pool
    cpu_count = os.cpu_count() or 1
    if num_workers is None or num_workers < 1:
        num_workers = max(1, cpu_count - 2) if cpu_count > 4 else cpu_count

    print("==================================================")
    print(" Amazon ML Challenge 2026 - Parallel Inference")
    print("==================================================")
    print(f" LightGBM Model:       {model_path}")
    print(f" Decision Threshold:   {threshold:.2f}")
    print(f" CPU Cores Available:  {cpu_count}")
    print(f" Worker Processes:     {num_workers}")
    print(f" Batch / Chunk Size:   {batch_size}")
    print(f" Max Candidates:       {max_candidates}")
    print("==================================================")

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    print(f"\n[1/3] Reading test Source 1 entities from {s1_path}...")

    s1_by_country = defaultdict(list)
    total_s1_records = 0

    with open(s1_path, "r", encoding="utf-8") as f:
        next(f)
        for row_index, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                s1_by_country[country].append((row_index, eid, name, addr))
                total_s1_records += 1

    print(f" Total S1 Entities: {total_s1_records:,}")
    for c, records in sorted(s1_by_country.items()):
        print(f"   - {c:10s}: {len(records):,} entities")

    # Allocate final results buffer (index-aligned for zero-overhead streaming)
    all_results = [None] * total_s1_records
    total_matches_count = 0
    total_singletons_count = 0

    countries_order = sorted(s1_by_country.keys(), key=lambda c: 0 if c == 'France' else (1 if c == 'US' else 2))

    for country in countries_order:
        country_start = time.time()
        s1_records = s1_by_country[country]
        print(f"\n[2/3] Processing Country: {country} ({len(s1_records):,} entities)")

        # Step 1: Pre-load and normalize all target records for this country in memory
        target_records = {}
        blocker = CountryBlocker(country, max_candidates=max_candidates, max_freq=2500)

        for fn in ["test_source2.tsv", "test_source3.tsv"]:
            fn_path = os.path.join(test_dir, fn)
            print(f"  Indexing targets from {fn} for {country}...")
            count = 0
            with open(fn_path, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 4 and parts[3] == country:
                        eid, name, addr = parts[0], parts[1], parts[2]
                        blocker.add_target_record(eid, name, addr)
                        target_records[eid] = {
                            "name": normalize_business_name(name),
                            "addr": normalize_address(addr, country),
                        }
                        count += 1
            print(f"    Indexed {count:,} records from {fn}.")

        print(f"  Total target records in memory: {len(target_records):,}")

        # Step 2: Slice S1 records into chunks
        chunks = [
            s1_records[i: i + batch_size]
            for i in range(0, len(s1_records), batch_size)
        ]
        print(f"  Dispatched {len(chunks)} chunks across {num_workers} parallel workers...")

        country_matches = 0
        country_singletons = 0
        processed_entities = 0

        # Step 3: Execute in parallel across worker processes
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_worker,
            initargs=(blocker, target_records, model_path, threshold, country),
        ) as executor:
            future_to_chunk = {
                executor.submit(process_s1_chunk, chunk): len(chunk)
                for chunk in chunks
            }

            for future in as_completed(future_to_chunk):
                chunk_len = future_to_chunk[future]
                chunk_res = future.result()

                for row_idx, s1_id, matches_str, cands_str in chunk_res:
                    all_results[row_idx] = (s1_id, matches_str, cands_str)
                    if matches_str:
                        country_matches += len(matches_str.split(","))
                    else:
                        country_singletons += 1

                processed_entities += chunk_len
                elapsed = time.time() - country_start
                rate = processed_entities / max(elapsed, 0.001)
                remaining = len(s1_records) - processed_entities
                eta_sec = remaining / max(rate, 1)

                if processed_entities % (batch_size * 5) == 0 or processed_entities == len(s1_records):
                    print(
                        f"    Progress: {processed_entities:,}/{len(s1_records):,} "
                        f"({rate:.1f} ent/s, ETA: {eta_sec:.0f}s) | "
                        f"Matches: {country_matches:,} | Singletons: {country_singletons:,}"
                    )

        total_matches_count += country_matches
        total_singletons_count += country_singletons
        print(f"  Finished {country} in {time.time()-country_start:.1f}s.")

        # Release country memory
        del target_records
        del blocker
        gc.collect()

    # Step 4: Stream final outputs in exact line-for-line order matching test_source1.tsv
    matching_file = os.path.join(output_dir, "matching_results.tsv")
    candidate_file = os.path.join(output_dir, "candidate_pairs.tsv")

    print(f"\n[3/3] Writing final submission files...")
    with open(matching_file, "w", encoding="utf-8") as f_m, \
         open(candidate_file, "w", encoding="utf-8") as f_c:

        f_m.write("source1_entity_id\tmatched_entity_ids\n")
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")

        for row_idx, res in enumerate(all_results):
            if res is None:
                raise RuntimeError(f"Missing prediction at row {row_idx}! Pipeline integrity violated.")
            s1_id, matches_str, cands_str = res
            f_m.write(f"{s1_id}\t{matches_str}\n")
            f_c.write(f"{s1_id}\t{cands_str}\n")

    total_time = time.time() - start_time
    print("==================================================")
    print(f" Inference Finished Successfully in {total_time:.1f}s ({total_time/60:.1f} mins)!")
    print(f" Total Source 1 Entities:   {total_s1_records:,}")
    print(f" Total Matches Predicted:   {total_matches_count:,}")
    print(f" Total Singletons:          {total_singletons_count:,} ({total_singletons_count/total_s1_records*100:.2f}%)")
    print(f" Average Matches / Entity:  {total_matches_count/total_s1_records:.2f}")
    print(f" Average Throughput:        {total_s1_records/max(total_time, 0.001):.1f} ent/s")
    print(f" Outputs:")
    print(f"   - {matching_file}")
    print(f"   - {candidate_file}")
    print("==================================================")

    # Automated Validation Check
    validator_script = os.path.join(test_dir, "../../utils/validate_submission.py")
    if os.path.exists(validator_script):
        print("\nRunning official submission validator...")
        os.system(
            f"{sys.executable} {validator_script} "
            f"--matching {matching_file} "
            f"--candidate {candidate_file} "
            f"--test-dir {test_dir}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Throughput Business Entity Resolution Inference Pipeline.")
    parser.add_argument("--test-dir", default="DATA/student_resource/dataset/test",
                        help="Path to test dataset directory.")
    parser.add_argument("--output-dir", default="output",
                        help="Directory to save final TSV files.")
    parser.add_argument("--model-path", default=None,
                        help="Path to trained LightGBM model file.")
    parser.add_argument("--meta-path", default=None,
                        help="Path to model metadata file.")
    parser.add_argument("--batch-size", type=int, default=5000,
                        help="Number of S1 records per worker task chunk.")
    parser.add_argument("--max-candidates", type=int, default=25,
                        help="Maximum candidate target records retained per S1 entity.")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of worker processes. Default: max(1, CPU cores - 2).")
    parser.add_argument("--threshold", type=float, default=0.69,
                        help="Decision threshold overriding metadata.")

    args = parser.parse_args()
    run_pipeline(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        meta_path=args.meta_path,
        threshold=args.threshold,
        batch_size=args.batch_size,
        max_candidates=args.max_candidates,
        num_workers=args.num_workers,
    )
