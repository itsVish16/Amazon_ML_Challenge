"""
Ultra-Fast & Multi-Tiered Entity Resolution Inference Pipeline.
Implements:
  1. Microsecond early-exit filtering on disjoint house numbers and name tokens.
  2. Multi-Tiered Decision Engine:
     - Tier 1: Deterministic dual-anchor & domain matches.
     - Tier 2: High-confidence LightGBM scoring with conflict-free constraints.
     - Tier 3: Strict conflict-aware pruning (eradicates false merges).
  3. EXACT row-for-row alignment matching test_source1.tsv.
  4. Memory-safe country partitioning (< 3 GB peak RAM).
"""

import os
import sys
import time
import pickle
import sqlite3
import argparse
import numpy as np
import lightgbm as lgb
from collections import defaultdict

from normalization import normalize_business_name, normalize_address
from blocking import CountryBlocker
from features import compute_pair_features, FEATURE_NAMES
from decision import should_accept_pair, should_skip_pair


def load_batch_targets(target_db: sqlite3.Connection, candidate_ids: set, country: str) -> dict:
    """Read and normalise only targets needed by one S1 batch.

    Keeping every normalised S2/S3 record in a Python dictionary is what made
    large countries exceed RAM.  The blocker stays in memory, while this
    bounded cache is rebuilt for each batch.
    """
    records = {}
    candidate_ids = list(candidate_ids)
    for start in range(0, len(candidate_ids), 900):
        chunk = candidate_ids[start:start + 900]
        placeholders = ",".join("?" for _ in chunk)
        query = f"SELECT entity_id, business_name, business_address FROM targets WHERE entity_id IN ({placeholders})"
        for entity_id, name, address in target_db.execute(query, chunk):
            records[entity_id] = {
                "name": normalize_business_name(name),
                "addr": normalize_address(address, country),
            }
    return records


def run_pipeline(
    test_dir: str = "DATA/student_resource/dataset/test",
    output_dir: str = "output",
    model_path: str = None,
    meta_path: str = None,
    threshold: float = 0.70,
    batch_size: int = 1000,
    max_candidates: int = 25,
):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    src_dir = os.path.dirname(__file__)
    if model_path is None:
        model_path = os.path.join(src_dir, "matching_model.txt")
    if meta_path is None:
        meta_path = os.path.join(src_dir, "model_meta.pkl")

    print(f"Loading LightGBM model from {model_path}...")
    model = lgb.Booster(model_file=model_path)

    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
            threshold = meta.get("threshold", threshold)
            expected_features = meta.get("feature_names")
            if expected_features != FEATURE_NAMES:
                raise RuntimeError(
                    "Model metadata does not match the active feature contract. "
                    "Retrain with src/train.py before inference."
                )
    if model.num_feature() != len(FEATURE_NAMES) or model.feature_name() != FEATURE_NAMES:
        raise RuntimeError(
            "LightGBM model does not match the active feature contract. "
            "Retrain with src/train.py before inference."
        )
    print(f"Using Base Decision Threshold: {threshold:.2f}")

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    print(f"Reading test entities from {s1_path}...")

    s1_by_country = defaultdict(list)

    with open(s1_path, "r", encoding="utf-8") as f:
        next(f)
        for row_index, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                eid = parts[0]
                s1_by_country[parts[3]].append((row_index, eid, parts[1], parts[2]))

    print(f"Total S1 entities: {sum(len(records) for records in s1_by_country.values())}")
    print(f"Country breakdown: { {k: len(v) for k, v in s1_by_country.items()} }")

    # Results are hundreds of MB.  Keeping them in Python dictionaries along
    # with an active country index causes avoidable memory pressure.  SQLite is
    # used only as a temporary ordered spool and is deleted after final TSVs
    # have been streamed.
    cache_path = os.path.join(output_dir, ".inference_results.sqlite3")
    if os.path.exists(cache_path):
        os.remove(cache_path)
    result_db = sqlite3.connect(cache_path)
    result_db.execute("PRAGMA journal_mode=OFF")
    result_db.execute("PRAGMA synchronous=OFF")
    result_db.execute(
        "CREATE TABLE results (row_index INTEGER PRIMARY KEY, source_id TEXT, matches TEXT, candidates TEXT)"
    )

    countries_order = sorted(s1_by_country.keys(), key=lambda c: 0 if c == 'France' else (1 if c == 'US' else 2))

    total_matches_count = 0
    total_singletons_count = 0

    for country in countries_order:
        country_start = time.time()
        s1_records = s1_by_country[country]
        print(f"\n==================================================")
        print(f" Processing Country: {country} ({len(s1_records)} entities)")
        print(f"==================================================")

        # 1. Index all targets.  Raw target fields are spooled to a temporary
        # SQLite table; only candidates from the active S1 batch are held as
        # normalised Python objects.
        blocker = CountryBlocker(country, max_candidates=max_candidates, max_freq=2500)
        target_cache_path = os.path.join(output_dir, f".targets_{country.lower()}.sqlite3")
        if os.path.exists(target_cache_path):
            os.remove(target_cache_path)
        target_db = sqlite3.connect(target_cache_path)
        target_db.execute("PRAGMA journal_mode=OFF")
        target_db.execute("PRAGMA synchronous=OFF")
        target_db.execute("CREATE TABLE targets (entity_id TEXT PRIMARY KEY, business_name TEXT, business_address TEXT)")

        for fn in ["test_source2.tsv", "test_source3.tsv"]:
            fn_path = os.path.join(test_dir, fn)
            print(f"Indexing {fn} for {country}...")
            count = 0
            insert_rows = []
            with open(fn_path, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 4 and parts[3] == country:
                        eid = parts[0]
                        name = parts[1]
                        addr = parts[2]
                        blocker.add_target_record(eid, name, addr)
                        insert_rows.append((eid, name, addr))
                        if len(insert_rows) >= 10000:
                            target_db.executemany(
                                "INSERT INTO targets (entity_id, business_name, business_address) VALUES (?, ?, ?)", insert_rows
                            )
                            insert_rows.clear()
                        count += 1
            if insert_rows:
                target_db.executemany(
                    "INSERT INTO targets (entity_id, business_name, business_address) VALUES (?, ?, ?)", insert_rows
                )
            target_db.commit()
            print(f"  Indexed {count} records from {fn}.")

        target_count = target_db.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
        print(f"Total target records indexed for {country}: {target_count}")

        # 2. Batch score candidates with early-exit and multi-tiered decision rules
        country_matches = 0
        country_singletons = 0

        for b_idx in range(0, len(s1_records), batch_size):
            batch_records = s1_records[b_idx: b_idx + batch_size]

            batch_pair_features = []
            batch_pair_indices = []
            batch_candidates = []
            batch_s1_norms = []
            batch_candidate_ids = set()

            for rec_i, (_, s1_id, s1_name, s1_addr) in enumerate(batch_records):
                s1_n = normalize_business_name(s1_name)
                s1_a = normalize_address(s1_addr, country)
                s1_norm = {
                    'name': s1_n,
                    'addr': s1_a,
                    'country': country,
                }
                batch_s1_norms.append(s1_norm)

                cands = blocker.retrieve_candidates(s1_name, s1_addr)
                batch_candidates.append(cands)
                batch_candidate_ids.update(cands)

            target_records = load_batch_targets(target_db, batch_candidate_ids, country)

            for rec_i, cands in enumerate(batch_candidates):
                s1_norm = batch_s1_norms[rec_i]
                for cid in cands:
                    if cid in target_records:
                        cn = target_records[cid]

                        if should_skip_pair(s1_norm, cn):
                            continue

                        feat = compute_pair_features(s1_norm, cn, cid)
                        batch_pair_features.append(feat)
                        batch_pair_indices.append((rec_i, cid, cn))

            # Batch inference with LightGBM
            matched_dict = defaultdict(list)
            if batch_pair_features:
                X_batch = np.array(batch_pair_features, dtype=np.float32)
                probs = model.predict(X_batch)

                for (rec_i, cid, cn), prob in zip(batch_pair_indices, probs):
                    if should_accept_pair(batch_s1_norms[rec_i], cn, prob, threshold):
                        matched_dict[rec_i].append(cid)

            # Store predictions
            result_rows = []
            for rec_i, (row_index, s1_id, _, _) in enumerate(batch_records):
                cands = batch_candidates[rec_i]
                matches = matched_dict.get(rec_i, [])

                all_cands = sorted(set(cands) | set(matches))
                result_rows.append((row_index, s1_id, ",".join(matches), ",".join(all_cands)))

                if matches:
                    country_matches += len(matches)
                else:
                    country_singletons += 1

            result_db.executemany(
                "INSERT INTO results (row_index, source_id, matches, candidates) VALUES (?, ?, ?, ?)", result_rows
            )
            result_db.commit()

            processed = b_idx + len(batch_records)
            if processed % 100000 == 0 or processed == len(s1_records):
                elapsed = time.time() - country_start
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"  Processed {processed}/{len(s1_records)} ({rate:.1f} ent/s). "
                      f"Singletons: {country_singletons}")

        total_matches_count += country_matches
        total_singletons_count += country_singletons
        print(f"Completed {country} in {time.time()-country_start:.1f}s.")

        del target_records
        del blocker
        target_db.close()
        os.remove(target_cache_path)

    # 3. Stream outputs in exact original test_source1.tsv order.
    matching_file = os.path.join(output_dir, "matching_results.tsv")
    candidate_file = os.path.join(output_dir, "candidate_pairs.tsv")

    print(f"\nWriting final outputs in exact original test_source1.tsv order...")
    with open(matching_file, "w", encoding="utf-8") as f_m, \
         open(candidate_file, "w", encoding="utf-8") as f_c:

        f_m.write("source1_entity_id\tmatched_entity_ids\n")
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")

        for s1_id, matches, candidates in result_db.execute(
            "SELECT source_id, matches, candidates FROM results ORDER BY row_index"
        ):
            f_m.write(f"{s1_id}\t{matches}\n")
            f_c.write(f"{s1_id}\t{candidates}\n")

    result_db.close()
    os.remove(cache_path)

    print(f"==================================================")
    print(f" Pipeline Finished Successfully in {time.time()-start_time:.1f}s!")
    print(f" Total Source 1 Entities: {sum(len(records) for records in s1_by_country.values())}")
    print(f" Total Matches Predicted: {total_matches_count}")
    total_source1 = sum(len(records) for records in s1_by_country.values())
    print(f" Total Singletons: {total_singletons_count} ({total_singletons_count/total_source1*100:.2f}%)")
    print(f" Average Matches per S1 Entity: {total_matches_count/total_source1:.2f}")
    print(f" Output files written to:")
    print(f"   - {matching_file}")
    print(f"   - {candidate_file}")
    print(f"==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run precision-first entity resolution inference.")
    parser.add_argument("--test-dir", default="DATA/student_resource/dataset/test")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--meta-path", default=None)
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="S1 records per scoring batch; raise only when RAM permits.")
    parser.add_argument("--max-candidates", type=int, default=25,
                        help="Maximum final-stage candidates per Source 1 record.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_candidates < 1:
        parser.error("--max-candidates must be positive")
    run_pipeline(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        meta_path=args.meta_path,
        batch_size=args.batch_size,
        max_candidates=args.max_candidates,
    )
