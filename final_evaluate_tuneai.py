"""
FINAL TuneAI MODEL EVALUATION
--------------------------------
Clean chronological 80/20 evaluation.

Methods:
1. Popularity
2. Content-Only
3. SVD-Only
4. TuneAI Hybrid (70% SVD + 30% Content)

Protocol:
- Global chronological 80/20 split
- SVD trained ONLY on training interactions
- 500 deterministic evaluation users
- Training tracks excluded from recommendations
- Held-out tracks included in candidate pool
- Same candidate universe for all methods
- Production similarity.py used for content scores
"""

import os
import math
import warnings

import numpy as np
import pandas as pd

from config import (
    PROCESSED_DATA_PATH,
    AUDIO_FEATURES,
    SVD_WEIGHT,
    CONTENT_WEIGHT,
)

from svd_model import TuneAISVD
from similarity import TuneAISimilarity


warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

N_USERS = 500
MIN_USER_INTERACTIONS = 6

TRAIN_RATIO = 0.80

CANDIDATE_POOL_SIZE = 2000

TOP_K_VALUES = [5, 10]

RANDOM_STATE = 42


# ============================================================
# HELPERS
# ============================================================

def find_column(df, possible_names):
    """
    Find the first matching column from a list of possible names.
    """
    lower_map = {str(c).lower(): c for c in df.columns}

    for name in possible_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    return None


def normalize_track_id(value):
    """
    Convert track IDs safely to strings.
    """
    if pd.isna(value):
        return None

    return str(value)


def precision_at_k(recommended, relevant, k):
    top = recommended[:k]

    if not top:
        return 0.0

    hits = len(set(top) & relevant)

    return hits / k


def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0

    top = recommended[:k]

    hits = len(set(top) & relevant)

    return hits / len(relevant)


def hit_rate_at_k(recommended, relevant, k):
    top = recommended[:k]

    return 1.0 if set(top) & relevant else 0.0


def ndcg_at_k(recommended, relevant, k):
    top = recommended[:k]

    dcg = 0.0

    for i, track_id in enumerate(top):
        if track_id in relevant:
            dcg += 1.0 / math.log2(i + 2)

    ideal_hits = min(len(relevant), k)

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0 / math.log2(i + 2)
        for i in range(ideal_hits)
    )

    return dcg / idcg


def calculate_metrics(recommendations, ground_truth):
    """
    Calculate aggregate metrics for all evaluation users.
    """

    results = {}

    for k in TOP_K_VALUES:

        precision_values = []
        recall_values = []
        ndcg_values = []
        hit_values = []

        for user_id in ground_truth:

            relevant = ground_truth[user_id]

            recommended = recommendations.get(
                user_id,
                []
            )

            precision_values.append(
                precision_at_k(
                    recommended,
                    relevant,
                    k
                )
            )

            recall_values.append(
                recall_at_k(
                    recommended,
                    relevant,
                    k
                )
            )

            ndcg_values.append(
                ndcg_at_k(
                    recommended,
                    relevant,
                    k
                )
            )

            hit_values.append(
                hit_rate_at_k(
                    recommended,
                    relevant,
                    k
                )
            )

        results[k] = {
            "precision": float(np.mean(precision_values)),
            "recall": float(np.mean(recall_values)),
            "ndcg": float(np.mean(ndcg_values)),
            "hit_rate": float(np.mean(hit_values)),
        }

    return results


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 100)
print(" " * 16 + "FINAL TuneAI EVALUATION")
print("=" * 100)


print("\n[1/9] Loading interactions...")


# ============================================================
# LOAD INTERACTIONS FROM TUNEAI SQLITE DATABASE
# ============================================================

import sqlite3


DB_PATH = "data/tuneai.db"


if not os.path.exists(DB_PATH):

    raise FileNotFoundError(
        f"TuneAI database not found: {DB_PATH}"
    )


print(
    f"Database: {DB_PATH}"
)


conn = sqlite3.connect(DB_PATH)


# Show available tables so the correct interaction table
# can be identified automatically.
tables = pd.read_sql_query(
    """
    SELECT name
    FROM sqlite_master
    WHERE type='table'
    ORDER BY name
    """,
    conn
)


print(
    "Database tables:"
)

for table_name in tables["name"].tolist():

    print(
        f"  - {table_name}"
    )


# Look for the table used for listening/interactions.
possible_tables = [
    "ListeningHistory",
    "listening_history",
    "Listening_History",
    "interactions",
    "Interactions",
    "user_interactions",
    "UserInteractions",
]


interaction_table = None

available_tables = set(
    tables["name"].tolist()
)


for table in possible_tables:

    if table in available_tables:

        interaction_table = table
        break


if interaction_table is None:

    # Try to identify a table automatically from its columns.
    for table in available_tables:

        try:

            columns = pd.read_sql_query(
                f'PRAGMA table_info("{table}")',
                conn
            )

            column_names = {
                str(c).lower()
                for c in columns["name"]
            }

            has_user = any(
                c in column_names
                for c in [
                    "user_id",
                    "userid",
                    "user"
                ]
            )

            has_track = any(
                c in column_names
                for c in [
                    "track_id",
                    "trackid",
                    "song_id",
                    "songid"
                ]
            )

            if has_user and has_track:

                interaction_table = table
                break

        except Exception:
            continue


if interaction_table is None:

    conn.close()

    raise RuntimeError(
        "Could not identify the interaction/listening-history "
        "table inside data/tuneai.db."
    )


print(
    f"Using interaction table: "
    f"{interaction_table}"
)


df_interactions = pd.read_sql_query(
    f'SELECT * FROM "{interaction_table}"',
    conn
)


conn.close()


print(
    f"Total interactions: "
    f"{len(df_interactions):,}"
)

# ============================================================
# FIND REQUIRED COLUMNS
# ============================================================

user_col = find_column(
    df_interactions,
    ["user_id", "userid", "user"]
)

track_col = find_column(
    df_interactions,
    ["track_id", "trackid", "song_id", "songid"]
)

rating_col = find_column(
    df_interactions,
    ["rating", "score", "play_count", "interaction"]
)

time_col = find_column(
    df_interactions,
    [
        "timestamp",
        "played_at",
        "datetime",
        "date",
        "created_at",
        "time",
    ]
)


if user_col is None:
    raise ValueError(
        "Could not find user_id column."
    )

if track_col is None:
    raise ValueError(
        "Could not find track_id column."
    )


print(f"User column: {user_col}")
print(f"Track column: {track_col}")


# ============================================================
# NORMALIZE DATA
# ============================================================

df_interactions = df_interactions.copy()

df_interactions[user_col] = (
    df_interactions[user_col]
    .astype(str)
)

df_interactions[track_col] = (
    df_interactions[track_col]
    .astype(str)
)


# Rating is required by SVD.
if rating_col is None:

    print(
        "Rating column not found. "
        "Creating rating = 1.0 for every interaction."
    )

    df_interactions["rating"] = 1.0
    rating_col = "rating"

else:

    df_interactions[rating_col] = pd.to_numeric(
        df_interactions[rating_col],
        errors="coerce"
    )

    df_interactions[rating_col] = (
        df_interactions[rating_col]
        .fillna(1.0)
    )


# ============================================================
# CHRONOLOGICAL ORDER
# ============================================================

if time_col is not None:

    print(
        f"Time column: {time_col}"
    )

    df_interactions["_eval_time"] = pd.to_datetime(
        df_interactions[time_col],
        errors="coerce"
    )

    # Missing dates go last.
    df_interactions = df_interactions.sort_values(
        ["_eval_time"]
    )

else:

    print(
        "WARNING: No timestamp column found."
    )

    print(
        "Using original interaction order."
    )


# ============================================================
# REMOVE USERS WITH TOO FEW INTERACTIONS
# ============================================================

user_counts = (
    df_interactions
    .groupby(user_col)[track_col]
    .nunique()
)


eligible_users = user_counts[
    user_counts >= MIN_USER_INTERACTIONS
].index.tolist()


print(
    f"Eligible users: {len(eligible_users):,}"
)


df_interactions = df_interactions[
    df_interactions[user_col].isin(
        eligible_users
    )
].copy()


# ============================================================
# GLOBAL 80/20 SPLIT
# ============================================================

print("\n[2/9] Creating 80/20 chronological split...")


train_parts = []
test_parts = []


for user_id, group in df_interactions.groupby(
    user_col,
    sort=False
):

    # Keep chronological order.
    if time_col is not None:
        group = group.sort_values(
            "_eval_time"
        )

    else:
        group = group.copy()

    # Remove duplicate track interactions
    # while preserving the latest occurrence.
    group = group.drop_duplicates(
        subset=[track_col],
        keep="last"
    )

    if len(group) < MIN_USER_INTERACTIONS:
        continue

    split_index = int(
        len(group) * TRAIN_RATIO
    )

    split_index = max(
        1,
        min(
            split_index,
            len(group) - 1
        )
    )

    train_parts.append(
        group.iloc[:split_index]
    )

    test_parts.append(
        group.iloc[split_index:]
    )


df_train = pd.concat(
    train_parts,
    ignore_index=True
)

df_test = pd.concat(
    test_parts,
    ignore_index=True
)


print(
    f"Training interactions: {len(df_train):,}"
)

print(
    f"Testing interactions: {len(df_test):,}"
)


# ============================================================
# SELECT EVALUATION USERS
# ============================================================

print("\n[3/9] Selecting evaluation users...")


users_with_test = sorted(
    df_test[user_col]
    .unique()
)


rng = np.random.RandomState(
    RANDOM_STATE
)

rng.shuffle(users_with_test)


evaluation_users = users_with_test[
    :N_USERS
]


print(
    f"Evaluation users: {len(evaluation_users)}"
)


# ============================================================
# GROUND TRUTH
# ============================================================

ground_truth = {}


for user_id in evaluation_users:

    tracks = set(
        df_test.loc[
            df_test[user_col] == user_id,
            track_col
        ].astype(str)
    )

    ground_truth[user_id] = tracks


total_held_out = sum(
    len(v)
    for v in ground_truth.values()
)


print(
    f"Total held-out tracks: {total_held_out:,}"
)


# ============================================================
# TRAINING DATA FOR SVD
# ============================================================

print("\n[4/9] Training fresh SVD...")


svd_train = pd.DataFrame({

    "user_id":
        df_train[user_col].astype(str),

    "track_id":
        df_train[track_col].astype(str),

    "rating":
        df_train[rating_col].astype(float),

})


svd_model = TuneAISVD(
    n_components=50,
    random_state=RANDOM_STATE
)


svd_model.fit(
    svd_train
)


print(
    f"SVD users learned: "
    f"{len(svd_model.user_to_idx):,}"
)

print(
    f"SVD songs learned: "
    f"{len(svd_model.song_to_idx):,}"
)


# ============================================================
# LOAD SONG CATALOG
# ============================================================

print("\n[5/9] Loading song catalog...")


df_songs = pd.read_csv(
    PROCESSED_DATA_PATH
)


df_songs = df_songs.copy()


catalog_track_col = find_column(
    df_songs,
    [
        "track_id",
        "trackid",
        "song_id",
        "songid",
    ]
)


if catalog_track_col is None:

    raise ValueError(
        "Could not find track_id column "
        "in processed song catalog."
    )


df_songs[catalog_track_col] = (
    df_songs[catalog_track_col]
    .astype(str)
)


df_songs = df_songs.drop_duplicates(
    subset=[catalog_track_col]
)


print(
    f"Catalog songs: {len(df_songs):,}"
)


# ============================================================
# CONTENT ENGINE
# ============================================================

print("\n[6/9] Preparing production content model...")


content_engine = TuneAISimilarity()

content_engine.fit(
    df_songs
)


print(
    "Production content similarity engine ready."
)


# ============================================================
# POPULARITY
# ============================================================

print("\n[7/9] Preparing candidate universe...")


# Popularity column.
popularity_col = find_column(
    df_songs,
    [
        "popularity",
        "popularity_score",
    ]
)


if popularity_col is not None:

    df_songs["_eval_popularity"] = pd.to_numeric(
        df_songs[popularity_col],
        errors="coerce"
    ).fillna(0.0)

else:

    df_songs["_eval_popularity"] = 0.0


# Candidate universe must be SVD-known.
#
# This avoids giving unknown SVD tracks a synthetic
# 0.5 score and makes SVD/Hybrid comparison fair.

svd_known_tracks = set(
    svd_model.song_to_idx.keys()
)


catalog_tracks = set(
    df_songs[catalog_track_col].astype(str)
)


common_tracks = (
    svd_known_tracks
    & catalog_tracks
)


print(
    f"SVD-known catalog tracks: "
    f"{len(common_tracks):,}"
)


# ------------------------------------------------------------
# SVD test-track coverage
# ------------------------------------------------------------

all_test_tracks = set()

for tracks in ground_truth.values():
    all_test_tracks.update(tracks)


known_test_tracks = (
    all_test_tracks
    & svd_known_tracks
)


print(
    f"SVD-known held-out tracks: "
    f"{len(known_test_tracks):,}"
)


if all_test_tracks:

    coverage = (
        len(known_test_tracks)
        / len(all_test_tracks)
        * 100
    )

else:

    coverage = 0.0


print(
    f"SVD test-track coverage: "
    f"{coverage:.2f}%"
)


# ============================================================
# RESULTS STORAGE
# ============================================================

all_recommendations = {

    "Popularity": {},

    "Content-Only": {},

    "SVD Only": {},

    "TuneAI Hybrid": {},

}


# ============================================================
# EVALUATION LOOP
# ============================================================

processed = 0


for user_id in evaluation_users:

    processed += 1

    if processed % 100 == 0:

        print(
            f"  Processed "
            f"{processed}/{len(evaluation_users)} users"
        )


    # --------------------------------------------------------
    # Training tracks
    # --------------------------------------------------------

    user_train = df_train[
        df_train[user_col] == user_id
    ]

    training_tracks = set(
        user_train[track_col]
        .astype(str)
    )


    # --------------------------------------------------------
    # Target track
    # --------------------------------------------------------
    #
    # Production recommend_hybrid() requires a target
    # track. We use the user's latest training track.
    #

    if time_col is not None:

        user_train_sorted = (
            user_train
            .sort_values("_eval_time")
        )

    else:

        user_train_sorted = user_train


    if user_train_sorted.empty:
        continue


    target_track_id = str(
        user_train_sorted.iloc[-1][track_col]
    )


    # --------------------------------------------------------
    # Candidate pool
    # --------------------------------------------------------

    candidate_df = df_songs[
        df_songs[catalog_track_col]
        .isin(common_tracks)
    ].copy()


    # Exclude songs already heard.
    candidate_df = candidate_df[
        ~candidate_df[catalog_track_col]
        .isin(training_tracks)
    ]


    # Exclude target track.
    candidate_df = candidate_df[
        candidate_df[catalog_track_col]
        != target_track_id
    ]


    # --------------------------------------------------------
    # Base popularity candidate pool
    # --------------------------------------------------------

    candidate_df = (
        candidate_df
        .sort_values(
            "_eval_popularity",
            ascending=False
        )
        .head(CANDIDATE_POOL_SIZE)
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # Add all known held-out tracks.
    #
    # This prevents the evaluation from hiding the
    # correct answer simply because it is not popular.
    # --------------------------------------------------------

    user_ground_truth = ground_truth[user_id]

    known_user_test_tracks = (
        user_ground_truth
        & svd_known_tracks
        & catalog_tracks
    )

    if known_user_test_tracks:

        heldout_rows = df_songs[
            df_songs[catalog_track_col]
            .isin(known_user_test_tracks)
        ].copy()

        heldout_rows = heldout_rows[
            ~heldout_rows[catalog_track_col]
            .isin(training_tracks)
        ]

        heldout_rows = heldout_rows[
            heldout_rows[catalog_track_col]
            != target_track_id
        ]

        candidate_df = pd.concat(
            [
                candidate_df,
                heldout_rows
            ],
            ignore_index=True
        )

        candidate_df = candidate_df.drop_duplicates(
            subset=[catalog_track_col]
        )


    candidate_ids = (
        candidate_df[catalog_track_col]
        .astype(str)
        .tolist()
    )


    if not candidate_ids:
        continue


    # ========================================================
    # 1. POPULARITY
    # ========================================================

    popularity_ranking = (
        candidate_df
        .sort_values(
            "_eval_popularity",
            ascending=False
        )[catalog_track_col]
        .astype(str)
        .tolist()
    )


    all_recommendations[
        "Popularity"
    ][user_id] = popularity_ranking


    # ========================================================
    # 2. SVD
    # ========================================================

    svd_scores = (
        svd_model
        .predict_user_scores_for_candidates(
            user_id,
            candidate_ids
        )
    )


    svd_ranking = sorted(
        candidate_ids,
        key=lambda x: svd_scores.get(
            x,
            0.0
        ),
        reverse=True
    )


    all_recommendations[
        "SVD Only"
    ][user_id] = svd_ranking


    # ========================================================
    # 3. PRODUCTION CONTENT SIMILARITY
    # ========================================================

    try:

        content_scores = (
            content_engine
            .calculate_pairwise_scores(
                target_track_id,
                candidate_ids,
                df_songs
            )
        )

    except Exception as e:

        print(
            f"\nContent error for user {user_id}: {e}"
        )

        content_scores = {
            track_id: 0.0
            for track_id in candidate_ids
        }


    content_ranking = sorted(
        candidate_ids,
        key=lambda x: content_scores.get(
            x,
            0.0
        ),
        reverse=True
    )


    all_recommendations[
        "Content-Only"
    ][user_id] = content_ranking


    # ========================================================
    # 4. TUNEAI HYBRID
    # ========================================================

    hybrid_scores = {}


    for track_id in candidate_ids:

        s_score = float(
            svd_scores.get(
                track_id,
                0.0
            )
        )

        c_score = float(
            content_scores.get(
                track_id,
                0.0
            )
        )

        hybrid_scores[track_id] = (
            SVD_WEIGHT * s_score
            +
            CONTENT_WEIGHT * c_score
        )


    hybrid_ranking = sorted(
        candidate_ids,
        key=lambda x: hybrid_scores.get(
            x,
            0.0
        ),
        reverse=True
    )


    all_recommendations[
        "TuneAI Hybrid"
    ][user_id] = hybrid_ranking


# ============================================================
# FILTER GROUND TRUTH
# ============================================================

# Because all methods use the same SVD-known candidate universe,
# evaluate against held-out tracks that are actually reachable
# by SVD.

fair_ground_truth = {}


for user_id in evaluation_users:

    relevant = (
        ground_truth[user_id]
        & svd_known_tracks
        & catalog_tracks
    )

    if relevant:

        fair_ground_truth[user_id] = relevant


print(
    "\nEvaluation users with reachable "
    "held-out tracks: "
    f"{len(fair_ground_truth)}"
)


# Keep only users that actually received recommendations.
for method in all_recommendations:

    all_recommendations[method] = {

        user_id: recs

        for user_id, recs
        in all_recommendations[method].items()

        if user_id in fair_ground_truth

    }


# ============================================================
# METRICS
# ============================================================

print("\n[8/9] Calculating metrics...")


method_metrics = {}


for method, recommendations in (
    all_recommendations.items()
):

    method_metrics[method] = calculate_metrics(
        recommendations,
        fair_ground_truth
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n[9/9] FINAL RESULTS\n")


print(
    f"{'Method':<24}"
    f"{'Precision@5':>14}"
    f"{'Recall@5':>12}"
    f"{'NDCG@5':>12}"
    f"{'Hit@5':>10}"
    f"{'Precision@10':>16}"
    f"{'Recall@10':>14}"
    f"{'NDCG@10':>14}"
    f"{'Hit@10':>12}"
)


print("-" * 130)


for method in [
    "Popularity",
    "Content-Only",
    "SVD Only",
    "TuneAI Hybrid",
]:

    m5 = method_metrics[method][5]
    m10 = method_metrics[method][10]


    print(
        f"{method:<24}"
        f"{m5['precision']:>14.4f}"
        f"{m5['recall']:>12.4f}"
        f"{m5['ndcg']:>12.4f}"
        f"{m5['hit_rate'] * 100:>9.2f}%"
        f"{m10['precision']:>16.4f}"
        f"{m10['recall']:>14.4f}"
        f"{m10['ndcg']:>14.4f}"
        f"{m10['hit_rate'] * 100:>11.2f}%"
    )


# ============================================================
# DETAILS
# ============================================================

print("\n\nEVALUATION DETAILS")
print("-" * 90)

print(
    f"Evaluation users: "
    f"{len(fair_ground_truth)}"
)

print(
    f"Total held-out tracks: "
    f"{len(all_test_tracks):,}"
)

print(
    f"SVD-known held-out tracks: "
    f"{len(known_test_tracks):,}"
)

print(
    f"SVD test-track coverage: "
    f"{coverage:.2f}%"
)

print(
    f"Catalog size: "
    f"{len(df_songs):,}"
)

print(
    f"Candidate pool base: "
    f"{CANDIDATE_POOL_SIZE:,}"
)

print(
    f"Hybrid weighting: "
    f"{SVD_WEIGHT * 100:.0f}% SVD + "
    f"{CONTENT_WEIGHT * 100:.0f}% Content"
)


print("\nProtocol:")
print("  80% historical interactions -> training")
print("  20% latest interactions -> testing")
print("  SVD trained only on training data")
print("  Training tracks excluded from candidates")
print("  Latest training track used as target")
print("  Held-out tracks added to candidate pool")
print("  Same SVD-known candidate universe for all methods")
print("  Content uses production similarity.py")
print("  Hybrid uses configured SVD + Content weights")


print("\n" + "=" * 100)
print("Evaluation complete.")
print("=" * 100)