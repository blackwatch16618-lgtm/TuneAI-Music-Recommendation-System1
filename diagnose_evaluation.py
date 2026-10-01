import numpy as np
import pandas as pd

from config import RANDOM_STATE
from database import SessionLocal, ListeningHistory
from svd_model import TuneAISVD
from recommender import TuneAIRecommender


# ============================================================
# SETTINGS
# ============================================================

EVALUATION_USERS = 500
CANDIDATE_POOL_SIZE = 2000
HYBRID_LIMIT = 500


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("             TuneAI EVALUATION DIAGNOSTIC")
    print("=" * 78)

    rng = np.random.default_rng(RANDOM_STATE)

    # ========================================================
    # 1. LOAD INTERACTIONS
    # ========================================================

    print("\n[1/7] Loading interactions...")

    db = SessionLocal()

    try:

        rows = db.query(
            ListeningHistory.user_id,
            ListeningHistory.track_id,
            ListeningHistory.rating
        ).all()

    finally:

        db.close()

    df = pd.DataFrame(
        rows,
        columns=[
            "user_id",
            "track_id",
            "rating"
        ]
    )

    print(
        f"Total interactions: {len(df):,}"
    )

    # ========================================================
    # 2. GLOBAL 80/20 SPLIT
    # ========================================================

    print(
        "\n[2/7] Creating global 80/20 split..."
    )

    train_rows = []
    test_rows = []

    eligible_users = []

    for user_id, user_df in df.groupby(
        "user_id"
    ):

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

    df_train = pd.DataFrame(train_rows)
    df_test = pd.DataFrame(test_rows)

    print(
        f"Eligible users: {len(eligible_users):,}"
    )

    print(
        f"Training interactions: {len(df_train):,}"
    )

    print(
        f"Testing interactions: {len(df_test):,}"
    )

    # ========================================================
    # 3. SELECT USERS
    # ========================================================

    print(
        "\n[3/7] Selecting evaluation users..."
    )

    if len(eligible_users) > EVALUATION_USERS:

        evaluation_users = list(
            rng.choice(
                eligible_users,
                size=EVALUATION_USERS,
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
        "\n[4/7] Training fresh SVD..."
    )

    svd = TuneAISVD(
        n_components=50,
        random_state=RANDOM_STATE
    )

    svd.fit(df_train)

    print(
        f"SVD users: "
        f"{len(svd.user_to_idx):,}"
    )

    print(
        f"SVD songs: "
        f"{len(svd.song_to_idx):,}"
    )

    # ========================================================
    # 5. LOAD CATALOG / CONTENT ENGINE
    # ========================================================

    print(
        "\n[5/7] Loading catalog and content engine..."
    )

    recommender = (
        TuneAIRecommender.load_engine()
    )

    catalog = recommender.df_songs.copy()

    catalog_ids = set(
        catalog["track_id"]
        .dropna()
        .astype(str)
    )

    print(
        f"Catalog songs: "
        f"{len(catalog_ids):,}"
    )

    # ========================================================
    # 6. DIAGNOSTIC ANALYSIS
    # ========================================================

    print(
        "\n[6/7] Running diagnostics..."
    )

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

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total_test_tracks = 0
    svd_known_tracks = 0
    catalog_known_tracks = 0

    hybrid_removed_tracks = 0

    svd_min_values = []
    svd_max_values = []
    svd_mean_values = []

    content_min_values = []
    content_max_values = []
    content_mean_values = []

    hybrid_min_values = []
    hybrid_max_values = []
    hybrid_mean_values = []

    users_with_flat_svd = 0

    sample_count = 0

    # --------------------------------------------------------
    # Analyze users
    # --------------------------------------------------------

    for user_id in evaluation_users:

        if user_id not in train_by_user:
            continue

        if user_id not in test_by_user:
            continue

        train_items = train_by_user[
            user_id
        ]

        test_items = test_by_user[
            user_id
        ]

        total_test_tracks += len(
            test_items
        )

        for track_id in test_items:

            if track_id in svd.song_to_idx:

                svd_known_tracks += 1

            if track_id in catalog_ids:

                catalog_known_tracks += 1

        # ----------------------------------------------------
        # Candidate pool
        # ----------------------------------------------------

        if len(catalog_ids) > CANDIDATE_POOL_SIZE:

            candidates = set(
                rng.choice(
                    list(catalog_ids),
                    size=CANDIDATE_POOL_SIZE,
                    replace=False
                )
            )

        else:

            candidates = set(
                catalog_ids
            )

        candidates |= test_items

        candidates -= train_items

        candidate_list = list(
            candidates
        )

        # ----------------------------------------------------
        # SVD
        # ----------------------------------------------------

        svd_scores = (
            svd.predict_user_scores_for_candidates(
                user_id,
                candidate_list
            )
        )

        if svd_scores:

            values = np.array(
                list(svd_scores.values()),
                dtype=float
            )

            svd_min_values.append(
                values.min()
            )

            svd_max_values.append(
                values.max()
            )

            svd_mean_values.append(
                values.mean()
            )

            if np.ptp(values) < 0.000001:

                users_with_flat_svd += 1

        # ----------------------------------------------------
        # Content
        # ----------------------------------------------------

        target_track = next(
            iter(train_items)
        )

        try:

            content_scores = (
                recommender.similarity
                .calculate_pairwise_scores(
                    target_track,
                    candidate_list,
                    catalog
                )
            )

            if content_scores:

                values = np.array(
                    list(content_scores.values()),
                    dtype=float
                )

                content_min_values.append(
                    values.min()
                )

                content_max_values.append(
                    values.max()
                )

                content_mean_values.append(
                    values.mean()
                )

        except Exception as e:

            print(
                f"Content diagnostic error "
                f"for {user_id}: {e}"
            )

            content_scores = {}

        # ----------------------------------------------------
        # Hybrid 500 restriction
        # ----------------------------------------------------

        hybrid_df = catalog[
            catalog["track_id"]
            .astype(str)
            .isin(candidate_list)
        ].copy()

        before_count = len(
            hybrid_df
        )

        if before_count > HYBRID_LIMIT:

            hybrid_df = (
                hybrid_df
                .sort_values(
                    "popularity",
                    ascending=False
                )
                .head(HYBRID_LIMIT)
            )

        after_count = len(
            hybrid_df
        )

        selected_ids = set(
            hybrid_df["track_id"]
            .astype(str)
        )

        removed_test = (
            test_items
            - selected_ids
        )

        hybrid_removed_tracks += len(
            removed_test
        )

        # ----------------------------------------------------
        # Hybrid score distribution
        # ----------------------------------------------------

        hybrid_scores = {}

        for track_id in selected_ids:

            s_score = svd_scores.get(
                track_id,
                0.5
            )

            c_score = content_scores.get(
                track_id,
                0.5
            )

            hybrid_scores[track_id] = (
                0.70 * s_score
                +
                0.30 * c_score
            )

        if hybrid_scores:

            values = np.array(
                list(hybrid_scores.values()),
                dtype=float
            )

            hybrid_min_values.append(
                values.min()
            )

            hybrid_max_values.append(
                values.max()
            )

            hybrid_mean_values.append(
                values.mean()
            )

        sample_count += 1

        # ----------------------------------------------------
        # Print first 5 users
        # ----------------------------------------------------

        if sample_count <= 5:

            print("\n" + "-" * 78)

            print(
                f"USER: {user_id}"
            )

            print(
                f"Training tracks: "
                f"{len(train_items)}"
            )

            print(
                f"Test tracks: "
                f"{len(test_items)}"
            )

            print(
                f"Candidates: "
                f"{len(candidate_list)}"
            )

            print(
                f"Hybrid candidates: "
                f"{after_count}"
            )

            print(
                f"Test tracks removed by "
                f"500-limit: "
                f"{len(removed_test)}"
            )

            if svd_scores:

                print(
                    "SVD: "
                    f"min={min(svd_scores.values()):.4f}, "
                    f"max={max(svd_scores.values()):.4f}, "
                    f"mean={np.mean(list(svd_scores.values())):.4f}"
                )

            if content_scores:

                print(
                    "Content: "
                    f"min={min(content_scores.values()):.4f}, "
                    f"max={max(content_scores.values()):.4f}, "
                    f"mean={np.mean(list(content_scores.values())):.4f}"
                )

            if hybrid_scores:

                print(
                    "Hybrid: "
                    f"min={min(hybrid_scores.values()):.4f}, "
                    f"max={max(hybrid_scores.values()):.4f}, "
                    f"mean={np.mean(list(hybrid_scores.values())):.4f}"
                )

    # ========================================================
    # 7. SUMMARY
    # ========================================================

    print(
        "\n[7/7] DIAGNOSTIC SUMMARY"
    )

    print("=" * 78)

    print(
        f"Total held-out test tracks: "
        f"{total_test_tracks:,}"
    )

    print(
        f"Test tracks known by SVD: "
        f"{svd_known_tracks:,}"
    )

    if total_test_tracks:

        print(
            f"SVD test-track coverage: "
            f"{100 * svd_known_tracks / total_test_tracks:.2f}%"
        )

    print(
        f"Test tracks present in catalog: "
        f"{catalog_known_tracks:,}"
    )

    if total_test_tracks:

        print(
            f"Catalog test-track coverage: "
            f"{100 * catalog_known_tracks / total_test_tracks:.2f}%"
        )

    print()

    print(
        f"Test tracks removed by hybrid "
        f"500-song limit: "
        f"{hybrid_removed_tracks:,}"
    )

    if total_test_tracks:

        print(
            f"Hybrid candidate loss: "
            f"{100 * hybrid_removed_tracks / total_test_tracks:.2f}%"
        )

    print()

    if svd_min_values:

        print(
            "SVD score distribution across users:"
        )

        print(
            f"  Average minimum: "
            f"{np.mean(svd_min_values):.4f}"
        )

        print(
            f"  Average maximum: "
            f"{np.mean(svd_max_values):.4f}"
        )

        print(
            f"  Average mean: "
            f"{np.mean(svd_mean_values):.4f}"
        )

        print(
            f"  Users with completely flat SVD: "
            f"{users_with_flat_svd}/{sample_count}"
        )

    print()

    if content_min_values:

        print(
            "Content score distribution:"
        )

        print(
            f"  Average minimum: "
            f"{np.mean(content_min_values):.4f}"
        )

        print(
            f"  Average maximum: "
            f"{np.mean(content_max_values):.4f}"
        )

        print(
            f"  Average mean: "
            f"{np.mean(content_mean_values):.4f}"
        )

    print()

    if hybrid_min_values:

        print(
            "Hybrid score distribution:"
        )

        print(
            f"  Average minimum: "
            f"{np.mean(hybrid_min_values):.4f}"
        )

        print(
            f"  Average maximum: "
            f"{np.mean(hybrid_max_values):.4f}"
        )

        print(
            f"  Average mean: "
            f"{np.mean(hybrid_mean_values):.4f}"
        )

    print("=" * 78)
    print(
        "Diagnostic completed."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()