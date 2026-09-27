"""Training and production-faithful calibration for entity resolution.

Key improvements over previous version:
  - Full-target training (6M+ records) for real hard-negative exposure
  - 800 boosting rounds with deeper trees
  - Ranked matching validation using select_matches()
  - 24-feature expanded feature vector
"""

import argparse
import os
import pickle
import random
import zlib
from collections import defaultdict

import lightgbm as lgb
import numpy as np

from blocking import CountryBlocker
from decision import should_accept_pair, should_skip_pair, select_matches
from features import FEATURE_NAMES, compute_pair_features
from normalization import normalize_address, normalize_business_name


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


def _read_s1_reservoir(path: str, sample_size: int, seed: int) -> dict:
    """Uniform deterministic sample without loading the complete S1 table."""
    rng, reservoir = random.Random(seed), []
    with open(path, "r", encoding="utf-8") as handle:
        next(handle)
        for index, line in enumerate(handle):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            record = (parts[0], parts[1], parts[2], parts[3])
            if len(reservoir) < sample_size:
                reservoir.append(record)
            else:
                replacement = rng.randrange(index + 1)
                if replacement < sample_size:
                    reservoir[replacement] = record
    return {entity_id: (name, address, country) for entity_id, name, address, country in reservoir}


def _read_labels(path: str, selected_ids: set) -> tuple[dict, set]:
    labels, positive_ids = {}, set()
    with open(path, "r", encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if not parts or parts[0] not in selected_ids:
                continue
            matched = set(parts[1].split(",")) if len(parts) > 1 and parts[1] else set()
            labels[parts[0]] = matched
            positive_ids.update(matched)
    return labels, positive_ids


def _read_target_pool(data_dir: str, positive_ids: set, background_samples: int, seed: int) -> dict:
    """All sampled positives plus representative S2/S3-country background (or complete targets if background_samples <= 0)."""
    if background_samples <= 0:
        targets = {}
        for filename in ("train_source2.tsv", "train_source3.tsv"):
            path = os.path.join(data_dir, filename)
            print(f"  Loading full target universe from {filename}...")
            count = 0
            with open(path, "r", encoding="utf-8") as handle:
                next(handle)
                for line in handle:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 4:
                        targets[parts[0]] = (parts[1], parts[2], parts[3])
                        count += 1
            print(f"    Loaded {count:,} records.")
        print(f"  Total target pool: {len(targets):,}")
        return targets

    rng = random.Random(seed)
    targets, reservoirs, seen = {}, defaultdict(list), defaultdict(int)
    per_group_cap = max(1, background_samples // 4)
    for filename in ("train_source2.tsv", "train_source3.tsv"):
        source = filename.split("_")[1]
        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as handle:
            next(handle)
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                entity_id, name, address, country = parts[:4]
                if entity_id in positive_ids:
                    targets[entity_id] = (name, address, country)
                    continue
                group = (source, country)
                seen[group] += 1
                reservoir = reservoirs[group]
                record = (entity_id, name, address, country)
                if len(reservoir) < per_group_cap:
                    reservoir.append(record)
                else:
                    replacement = rng.randrange(seen[group])
                    if replacement < per_group_cap:
                        reservoir[replacement] = record
    for reservoir in reservoirs.values():
        for entity_id, name, address, country in reservoir:
            targets.setdefault(entity_id, (name, address, country))
    return targets


def _normalise_records(records: dict) -> dict:
    return {
        entity_id: {
            "name": normalize_business_name(name),
            "addr": normalize_address(address, country),
            "country": country,
        }
        for entity_id, (name, address, country) in records.items()
    }


def train_matching_model(
    data_dir: str = "DATA/student_resource/dataset/train",
    s1_samples: int = 50000,
    background_samples: int = 200000,
    seed: int = 20260927,
):
    print("=" * 60)
    print(" Training Entity Resolution Model")
    print("=" * 60)
    full_target_mode = background_samples <= 0
    print(f"  Mode: {'FULL-TARGET HARD-NEGATIVE' if full_target_mode else 'SAMPLED BACKGROUND'}")
    print(f"  S1 Samples:         {s1_samples:,}")
    print(f"  Background Samples: {'ALL (full universe)' if full_target_mode else f'{background_samples:,}'}")
    print(f"  Feature Count:      {len(FEATURE_NAMES)}")
    print("=" * 60)

    print(f"\n[1/6] Sampling {s1_samples:,} S1 entities uniformly from training data...")
    s1_data = _read_s1_reservoir(os.path.join(data_dir, "train_source1.tsv"), s1_samples, seed)
    labels, positive_ids = _read_labels(os.path.join(data_dir, "train_ground_truth.tsv"), set(s1_data))
    if len(labels) != len(s1_data):
        raise RuntimeError("Ground truth is missing selected Source 1 IDs.")
    singleton_count = sum(not matches for matches in labels.values())
    print(f"  Selected S1: {len(s1_data):,}; positives: {len(positive_ids):,}; "
          f"singletons: {singleton_count:,} ({100 * singleton_count / len(s1_data):.1f}%)")

    print(f"\n[2/6] Loading target records...")
    target_data = _read_target_pool(data_dir, positive_ids, background_samples, seed + 1)
    print(f"  Target training pool: {len(target_data):,}")

    print(f"\n[3/6] Normalizing records...")
    s1_norm = _normalise_records(s1_data)
    print(f"  Normalized {len(s1_norm):,} S1 records.")
    target_norm = _normalise_records(target_data)
    print(f"  Normalized {len(target_norm):,} target records.")

    print(f"\n[4/6] Building blocking indexes and generating pairs...")
    blockers = {
        c: CountryBlocker(c, max_candidates=25, max_freq=2500)
        for c in ("France", "US", "India")
    }
    for entity_id, (name, address, country) in target_data.items():
        if country in blockers:
            blockers[country].add_target_record(entity_id, name, address)
    for c, blocker in blockers.items():
        print(f"  {c}: {len(blocker.index):,} index keys")

    # Train/Validation split
    train_ids, validation_ids = [], []
    for entity_id in s1_data:
        split_value = zlib.crc32(entity_id.encode("utf-8")) & 0xFF
        (train_ids if split_value < 205 else validation_ids).append(entity_id)
    print(f"  Split: {len(train_ids):,} train / {len(validation_ids):,} validation S1 entities")

    # Generate training pairs
    train_features, train_labels = [], []
    skipped = 0
    for i, source_id in enumerate(train_ids):
        name, address, country = s1_data[source_id]
        candidates = set(blockers[country].retrieve_candidates(name, address))
        # Force-include all true positives that exist in target pool
        candidates.update(labels[source_id] & target_norm.keys())
        for candidate_id in candidates:
            candidate = target_norm.get(candidate_id)
            if candidate is None:
                continue
            if should_skip_pair(s1_norm[source_id], candidate):
                skipped += 1
                continue
            train_features.append(compute_pair_features(s1_norm[source_id], candidate, candidate_id))
            train_labels.append(int(candidate_id in labels[source_id]))
        if (i + 1) % 5000 == 0:
            print(f"    Processed {i + 1:,}/{len(train_ids):,} train entities "
                  f"({len(train_features):,} pairs, {sum(train_labels):,} positives)")

    X_train = np.asarray(train_features, dtype=np.float32)
    y_train = np.asarray(train_labels, dtype=np.int8)
    if not len(X_train) or not y_train.any() or y_train.all():
        raise RuntimeError("Training sample did not produce a usable mix of pairs.")
    neg_count = int(len(y_train) - y_train.sum())
    pos_count = int(y_train.sum())
    print(f"  Training pairs: {len(y_train):,}; positives: {pos_count:,}; "
          f"negatives: {neg_count:,}; skipped: {skipped:,}")
    print(f"  Negative:Positive ratio = {neg_count / max(pos_count, 1):.1f}:1")

    print(f"\n[5/6] Training LightGBM model (800 rounds)...")
    # Scale_pos_weight helps with imbalanced classes
    scale_pos = neg_count / max(pos_count, 1)
    model = lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 127,
            "max_depth": 10,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.80,
            "bagging_freq": 1,
            "min_data_in_leaf": 30,
            "scale_pos_weight": min(scale_pos, 20.0),
            "verbose": -1,
            "n_jobs": -1,
            "seed": seed,
        },
        lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES),
        num_boost_round=800,
    )
    print(f"  Model trained. {model.num_trees()} trees.")

    # Feature importance report
    importance = model.feature_importance(importance_type='gain')
    sorted_features = sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1])
    print(f"\n  Top-10 features by gain:")
    for feat, gain in sorted_features[:10]:
        print(f"    {feat:25s} {gain:12.1f}")

    # Validation
    print(f"\n[6/6] Validating with production-faithful inference...")
    validation_features, pair_locations, validation_rows = [], [], []
    true_links = retrieved_links = 0
    for row_index, source_id in enumerate(validation_ids):
        name, address, country = s1_data[source_id]
        candidates = blockers[country].retrieve_candidates(name, address)
        true_set = labels[source_id]
        true_links += len(true_set)
        retrieved_links += len(set(candidates) & true_set)
        for candidate_id in candidates:
            candidate = target_norm.get(candidate_id)
            if candidate is None or should_skip_pair(s1_norm[source_id], candidate):
                continue
            validation_features.append(compute_pair_features(s1_norm[source_id], candidate, candidate_id))
            pair_locations.append((row_index, candidate_id, candidate))
        validation_rows.append((source_id, true_set))

    probabilities = model.predict(np.asarray(validation_features, dtype=np.float32)) if validation_features else []
    per_row_scores = defaultdict(list)
    for (row_index, candidate_id, candidate), probability in zip(pair_locations, probabilities):
        per_row_scores[row_index].append((candidate_id, candidate, float(probability)))

    print(f"  Validation blocking recall: {retrieved_links / max(true_links, 1):.4%} "
          f"({retrieved_links:,}/{true_links:,})")

    # Sweep thresholds using ranked matching (select_matches)
    best_threshold, best_score = 0.80, -1.0
    print(f"\n  Sweeping thresholds 0.60 - 0.95 (using ranked select_matches)...")
    for threshold in np.arange(0.60, 0.951, 0.01):
        scores = []
        for row_index, (source_id, true_set) in enumerate(validation_rows):
            scored_candidates = per_row_scores[row_index]
            predicted = set(select_matches(
                s1_norm[source_id], scored_candidates, float(threshold)
            ))
            scores.append(compute_entity_f05(true_set, predicted))
        score = float(np.mean(scores))
        if score > best_score:
            best_threshold, best_score = float(threshold), score
        if int(threshold * 100) % 5 == 0:
            print(f"    Threshold {threshold:.2f}: Macro F0.5 = {score:.5f}")

    print(f"\n  *** Best production-faithful validation macro F0.5: {best_score:.5f} at threshold {best_threshold:.2f} ***")

    # Also evaluate with should_accept_pair for comparison
    best_sa_threshold, best_sa_score = 0.80, -1.0
    for threshold in np.arange(0.60, 0.951, 0.01):
        scores = []
        for row_index, (source_id, true_set) in enumerate(validation_rows):
            predicted = {
                candidate_id for candidate_id, candidate, probability in per_row_scores[row_index]
                if should_accept_pair(s1_norm[source_id], candidate, probability, float(threshold))
            }
            scores.append(compute_entity_f05(true_set, predicted))
        score = float(np.mean(scores))
        if score > best_sa_score:
            best_sa_threshold, best_sa_score = float(threshold), score
    print(f"  (Comparison) should_accept_pair best: {best_sa_score:.5f} at {best_sa_threshold:.2f}")

    # Use whichever method is better
    if best_sa_score > best_score:
        print(f"  NOTE: should_accept_pair performed better. Using its threshold.")
        best_threshold = best_sa_threshold
        best_score = best_sa_score

    source_dir = os.path.dirname(__file__)
    model_path = os.path.join(source_dir, "matching_model.txt")
    metadata_path = os.path.join(source_dir, "model_meta.pkl")
    model.save_model(model_path)
    with open(metadata_path, "wb") as handle:
        pickle.dump({
            "threshold": best_threshold,
            "feature_names": FEATURE_NAMES,
            "seed": seed,
            "s1_samples": s1_samples,
            "background_samples": background_samples,
            "validation_f05": best_score,
            "num_features": len(FEATURE_NAMES),
            "num_trees": model.num_trees(),
        }, handle)
    print(f"\n  Saved model to {model_path}")
    print(f"  Saved metadata to {metadata_path}")
    print(f"  Threshold: {best_threshold:.2f}")
    print(f"  Validation F0.5: {best_score:.5f}")
    return model, best_threshold, best_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train entity resolution model")
    parser.add_argument("--data-dir", default="DATA/student_resource/dataset/train")
    parser.add_argument("--s1-samples", type=int, default=50000)
    parser.add_argument("--background-samples", type=int, default=200000)
    parser.add_argument("--full-targets", action="store_true",
                        help="Index entire train target universe for hard-negative training (requires ~100GB+ RAM)")
    parser.add_argument("--seed", type=int, default=20260927)
    arguments = parser.parse_args()
    bg_samples = 0 if arguments.full_targets else arguments.background_samples
    train_matching_model(arguments.data_dir, arguments.s1_samples, bg_samples, arguments.seed)
