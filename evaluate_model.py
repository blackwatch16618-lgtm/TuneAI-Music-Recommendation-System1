"""
TuneAI - Final Academic Evaluation

Leakage-free evaluation:
- 80/20 split for ALL eligible users
- SVD trained only on training interactions
- 500 users evaluated
- Same 2,000-song candidate pool for every method
- No production 500-song popularity cutoff during evaluation

Methods:
    1. Popularity
    2. Content-Only
    3. SVD Only
    4. TuneAI Hybrid (70% SVD + 30% Content)

Metrics:
    Precision@5 / @10
    Recall@5 / @10
    NDCG@5 / @10
    Hit Rate@5 / @10
"""

import numpy as np
import pandas as pd

from config import RANDOM_STATE, SVD_WEIGHT, CONTENT_WEIGHT
from database import SessionLocal, ListeningHistory
from svd_model import TuneAISVD
from recommender import TuneAIRecommender


# ============================================================
# METRICS
# ============================================================

def precision_at_k(recommended, relevant, k):

    recommended = recommended[:k]
    relevant = set(relevant)

    if k == 0:
        return 0.0

    hits = len(
        set(recommended) & relevant
    )

    return hits / float(k)


def recall_at_k(recommended, relevant, k):

    recommended = recommended[:k]
    relevant = set(relevant)

    if not relevant:
        return 0.0

    hits = len(
        set(recommended) & relevant
    )

    return hits / float(len(relevant))


def ndcg_at_k(recommended, relevant, k):

    recommended = recommended[:k]
    relevant = set(relevant)

    dcg = 0.0

    for rank, track_id in enumerate(
        recommended,
        start=1
    ):

        if track_id in relevant:

            dcg += (
                1.0 /
                np.log2(rank + 1)
            )

    ideal_hits = min(
        len(relevant),
        k
    )

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0 /
        np.log2(rank + 1)
        for rank in range(
            1,
            ideal_hits + 1
        )
    )

    return dcg / idcg


def hit_rate_at_k(
    recommended,
    relevant,
    k
):

    recommended = set(
        recommended[:k]
    )

    relevant = set(relevant)

    return (
        1.0
        if recommended & relevant
        else 0.0
    )


# ============================================================
# MAIN
# ============================================================

def evaluate_model():

    print("=" * 82)
    print(
        "        TuneAI FINAL ACADEMIC EVALUATION"
    )
    print("=" * 82)

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    # ========================================================
    # 1. LOAD ALL INTERACTIONS
    # ========================================================

    print(
        "\n[1/8] Loading ALL interaction data..."
    )

    db = SessionLocal()

    try:

        rows = db.query(
            ListeningHistory.user_id,
            ListeningHistory.track_id,
            ListeningHistory.rating
        ).all()

    finally:

        db.close()

    df_all = pd.DataFrame(
        rows,
        columns=[
            "user_id",
            "track_id",
            "rating"
        ]
    )

    if df_all.empty:

        print(
            "ERROR: No interaction data found."
        )

        return

    print(
        f"Total interactions: "
        f"{len(df_all):,}"
    )

    # ========================================================
    # 2. 80/20 SPLIT FOR ALL USERS
    # ========================================================

    print(
        "\n[2/8] Creating 80/20 split "
        "for ALL users..."
    )

    train_rows = []
    test_rows = []

    eligible_users = []

    for user_id, user_df in df_all.groupby(
        "user_id"
    ):

        # Preserve the order returned by the database.
        tracks = list(
            dict.fromkeys(
                user_df["track_id"]
                .astype(str)
                .tolist()
            )
        )

        if len(tracks) < 6:
            continue

        eligible_users.append(user_id)

        n_test = max(
            1,
            int(len(tracks) * 0.20)
        )

        train_tracks = tracks[:-n_test]
        test_tracks = tracks[-n_test:]

        for track_id in train_tracks:

            row = user_df[
                user_df["track_id"].astype(str)
                == track_id
            ].iloc[0]

            train_rows.append({
                "user_id": user_id,
                "track_id": track_id,
                "rating": row["rating"]
            })

        for track_id in test_tracks:

            row = user_df[
                user_df["track_id"].astype(str)
                == track_id
            ].iloc[0]

            test_rows.append({
                "user_id": user_id,
                "track_id": track_id,
                "rating": row["rating"]
            })

    df_train = pd.DataFrame(
        train_rows
    )

    df_test = pd.DataFrame(
        test_rows
    )

    print(
        f"Eligible users: "
        f"{len(eligible_users):,}"
    )

    print(
        f"Training interactions: "
        f"{len(df_train):,}"
    )

    print(
        f"Testing interactions: "
        f"{len(df_test):,}"
    )

    # ========================================================
    # 3. SELECT 500 EVALUATION USERS
    # ========================================================

    print(
        "\n[3/8] Selecting evaluation users..."
    )

    if len(eligible_users) > 500:

        evaluation_users = list(
            rng.choice(
                eligible_users,
                size=500,
                replace=False
            )
        )

    else:

        evaluation_users = eligible_users

    print(
        f"Evaluation users: "
        f"{len(evaluation_users):,}"
    )

    # ========================================================
    # 4. TRAIN FRESH SVD
    # ========================================================

    print(
        "\n[4/8] Training fresh SVD "
        "on ALL training interactions..."
    )

    evaluation_svd = TuneAISVD(
        n_components=50,
        random_state=RANDOM_STATE
    )

    evaluation_svd.fit(
        df_train[
            [
                "user_id",
                "track_id",
                "rating"
            ]
        ]
    )

    print(
        "Fresh SVD training completed."
    )

    print(
        f"SVD components: "
        f"{evaluation_svd.n_components}"
    )

    print(
        f"SVD users learned: "
        f"{len(evaluation_svd.user_to_idx):,}"
    )

    print(
        f"SVD songs learned: "
        f"{len(evaluation_svd.song_to_idx):,}"
    )

    # ========================================================
    # 5. LOAD CONTENT ENGINE
    # ========================================================

    print(
        "\n[5/8] Loading TuneAI content engine..."
    )

    recommender = (
        TuneAIRecommender.load_engine()
    )

    catalog = recommender.df_songs.copy()

    catalog["track_id"] = (
        catalog["track_id"]
        .astype(str)
    )

    print(
        f"Catalog songs: "
        f"{len(catalog):,}"
    )

    print(
        f"Hybrid configuration: "
        f"{SVD_WEIGHT * 100:.0f}% SVD + "
        f"{CONTENT_WEIGHT * 100:.0f}% Content"
    )

    # ========================================================
    # 6. COMMON CANDIDATE POOL
    # ========================================================

    print(
        "\n[6/8] Preparing common candidate pool..."
    )

    catalog_ids = (
        catalog["track_id"]
        .dropna()
        .unique()
        .tolist()
    )

    if len(catalog_ids) > 2000:

        common_candidates = set(
            rng.choice(
                catalog_ids,
                size=2000,
                replace=False
            )
        )

    else:

        common_candidates = set(
            catalog_ids
        )

    print(
        f"Common candidate pool: "
        f"{len(common_candidates):,}"
    )

    # ========================================================
    # GROUP TRAIN / TEST DATA
    # ========================================================

    train_by_user = (
        df_train
        .groupby("user_id")["track_id"]
        .apply(
            lambda x: set(
                x.astype(str)
            )
        )
        .to_dict()
    )

    test_by_user = (
        df_test
        .groupby("user_id")["track_id"]
        .apply(
            lambda x: set(
                x.astype(str)
            )
        )
        .to_dict()
    )

    # ========================================================
    # RESULT STORAGE
    # ========================================================

    methods = [
        "Popularity",
        "Content-Only",
        "SVD Only",
        "TuneAI Hybrid"
    ]

    results = {}

    for method in methods:

        results[method] = {}

        for k in [5, 10]:

            results[method][k] = {
                "precision": [],
                "recall": [],
                "ndcg": [],
                "hit_rate": []
            }

    # ========================================================
    # EVALUATION LOOP
    # ========================================================

    print(
        "\n[7/8] Evaluating 500 held-out users..."
    )

    evaluated_users = 0
    svd_known_test_tracks = 0
    total_test_tracks = 0

    for user_id in evaluation_users:

        if user_id not in train_by_user:
            continue

        if user_id not in test_by_user:
            continue

        if user_id not in evaluation_svd.user_to_idx:
            continue

        train_items = train_by_user[
            user_id
        ]

        ground_truth = test_by_user[
            user_id
        ]

        if not ground_truth:
            continue

        total_test_tracks += len(
            ground_truth
        )

        # ----------------------------------------------------
        # Candidate pool
        #
        # Same candidates for EVERY method.
        # Held-out tracks are explicitly included.
        # Training tracks are excluded.
        # ----------------------------------------------------

        candidate_ids = (
            common_candidates
            | ground_truth
        )

        candidate_ids -= train_items

        candidate_ids = list(
            candidate_ids
        )

        if not candidate_ids:
            continue

        # ----------------------------------------------------
        # SVD coverage
        # ----------------------------------------------------

        for track_id in ground_truth:

            if track_id in evaluation_svd.song_to_idx:

                svd_known_test_tracks += 1

        # ----------------------------------------------------
        # Choose content seed
        # ----------------------------------------------------

        target_track = next(
            iter(train_items)
        )

        # ====================================================
        # POPULARITY
        # ====================================================

        popularity_df = catalog[
            catalog["track_id"].isin(
                candidate_ids
            )
        ].copy()

        popularity_df = (
            popularity_df
            .drop_duplicates(
                subset=["track_id"]
            )
            .sort_values(
                "popularity",
                ascending=False
            )
        )

        popularity_ranked = (
            popularity_df["track_id"]
            .tolist()
        )

        # ====================================================
        # CONTENT
        # ====================================================

        try:

            content_scores = (
                recommender.similarity
                .calculate_pairwise_scores(
                    target_track,
                    candidate_ids,
                    catalog
                )
            )

        except Exception:

            content_scores = {}

        content_ranked = sorted(
            content_scores.keys(),
            key=lambda track_id:
                content_scores[track_id],
            reverse=True
        )

        # ====================================================
        # SVD
        # ====================================================

        svd_scores = (
            evaluation_svd
            .predict_user_scores_for_candidates(
                user_id,
                candidate_ids
            )
        )

        svd_ranked = sorted(
            svd_scores.keys(),
            key=lambda track_id:
                svd_scores[track_id],
            reverse=True
        )

        # ====================================================
        # HYBRID
        # ====================================================

        hybrid_scores = {}

        for track_id in candidate_ids:

            s_score = svd_scores.get(
                track_id,
                0.5
            )

            c_score = content_scores.get(
                track_id,
                0.5
            )

            hybrid_scores[track_id] = (
                SVD_WEIGHT * s_score
                +
                CONTENT_WEIGHT * c_score
            )

        hybrid_ranked = sorted(
            hybrid_scores.keys(),
            key=lambda track_id:
                hybrid_scores[track_id],
            reverse=True
        )

        # ====================================================
        # CALCULATE METRICS
        # ====================================================

        recommendation_sets = {
            "Popularity": popularity_ranked,
            "Content-Only": content_ranked,
            "SVD Only": svd_ranked,
            "TuneAI Hybrid": hybrid_ranked
        }

        for method, ranked in (
            recommendation_sets.items()
        ):

            for k in [5, 10]:

                results[method][k][
                    "precision"
                ].append(
                    precision_at_k(
                        ranked,
                        ground_truth,
                        k
                    )
                )

                results[method][k][
                    "recall"
                ].append(
                    recall_at_k(
                        ranked,
                        ground_truth,
                        k
                    )
                )

                results[method][k][
                    "ndcg"
                ].append(
                    ndcg_at_k(
                        ranked,
                        ground_truth,
                        k
                    )
                )

                results[method][k][
                    "hit_rate"
                ].append(
                    hit_rate_at_k(
                        ranked,
                        ground_truth,
                        k
                    )
                )

        evaluated_users += 1

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print(
        "\n[8/8] FINAL ACADEMIC RESULTS"
    )

    print("=" * 112)

    print(
        f"{'Method':<20}"
        f"{'Precision@5':>15}"
        f"{'Recall@5':>13}"
        f"{'NDCG@5':>12}"
        f"{'Hit@5':>11}"
        f"{'Precision@10':>16}"
        f"{'Recall@10':>14}"
        f"{'NDCG@10':>13}"
        f"{'Hit@10':>12}"
    )

    print("-" * 112)

    for method in methods:

        m5 = results[method][5]
        m10 = results[method][10]

        p5 = np.mean(
            m5["precision"]
        )

        r5 = np.mean(
            m5["recall"]
        )

        n5 = np.mean(
            m5["ndcg"]
        )

        h5 = np.mean(
            m5["hit_rate"]
        )

        p10 = np.mean(
            m10["precision"]
        )

        r10 = np.mean(
            m10["recall"]
        )

        n10 = np.mean(
            m10["ndcg"]
        )

        h10 = np.mean(
            m10["hit_rate"]
        )

        print(
            f"{method:<20}"
            f"{p5:>15.4f}"
            f"{r5:>13.4f}"
            f"{n5:>12.4f}"
            f"{h5 * 100:>10.2f}%"
            f"{p10:>15.4f}"
            f"{r10:>13.4f}"
            f"{n10:>12.4f}"
            f"{h10 * 100:>10.2f}%"
        )

    print("=" * 112)

    # ========================================================
    # COVERAGE
    # ========================================================

    print(
        f"\nEvaluated users: "
        f"{evaluated_users:,}"
    )

    print(
        f"Total held-out test tracks: "
        f"{total_test_tracks:,}"
    )

    print(
        f"SVD-known held-out tracks: "
        f"{svd_known_test_tracks:,}"
    )

    if total_test_tracks > 0:

        print(
            f"SVD test-track coverage: "
            f"{100 * svd_known_test_tracks / total_test_tracks:.2f}%"
        )

    print(
        f"\nCandidate pool used by every method: "
        f"{len(common_candidates):,}"
    )

    print(
        "\nEvaluation protocol:"
    )

    print(
        "80% historical interactions -> training"
    )

    print(
        "20% latest interactions -> testing"
    )

    print(
        "SVD trained only on training interactions"
    )

    print(
        "Same candidate pool used for all methods"
    )

    print(
        f"Hybrid = "
        f"{SVD_WEIGHT * 100:.0f}% SVD + "
        f"{CONTENT_WEIGHT * 100:.0f}% Content"
    )

    print(
        "\nNo production 500-song cutoff "
        "was applied during academic evaluation."
    )

    print("=" * 82)

    print(
        "Evaluation completed successfully."
    )

    print("=" * 82)


if __name__ == "__main__":
    evaluate_model()