import numpy as np
import pandas as pd

from config import RANDOM_STATE, SVD_WEIGHT, CONTENT_WEIGHT
from database import SessionLocal, ListeningHistory
from svd_model import TuneAISVD
from recommender import TuneAIRecommender


def main():

    print("=" * 80)
    print("             TuneAI SCORE DEBUGGING")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. LOAD DATABASE
    # --------------------------------------------------------

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
        f"\nTotal interactions: {len(df):,}"
    )

    # --------------------------------------------------------
    # 2. SELECT USERS
    # --------------------------------------------------------

    user_counts = (
        df.groupby("user_id")["track_id"]
        .nunique()
    )

    eligible_users = user_counts[
        user_counts >= 6
    ].index.tolist()

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    selected_users = list(
        rng.choice(
            eligible_users,
            size=min(5, len(eligible_users)),
            replace=False
        )
    )

    print(
        f"Users selected: {len(selected_users)}"
    )

    # --------------------------------------------------------
    # 3. CREATE TRAIN / TEST SPLIT
    # --------------------------------------------------------

    train_rows = []
    test_rows = []

    for user_id in selected_users:

        user_df = df[
            df["user_id"] == user_id
        ].copy()

        tracks = list(
            dict.fromkeys(
                user_df["track_id"].tolist()
            )
        )

        if len(tracks) < 6:
            continue

        n_test = max(
            1,
            int(len(tracks) * 0.20)
        )

        train_tracks = tracks[:-n_test]
        test_tracks = tracks[-n_test:]

        for track_id in train_tracks:

            row = user_df[
                user_df["track_id"] == track_id
            ].iloc[0]

            train_rows.append({
                "user_id": user_id,
                "track_id": track_id,
                "rating": row["rating"]
            })

        for track_id in test_tracks:

            row = user_df[
                user_df["track_id"] == track_id
            ].iloc[0]

            test_rows.append({
                "user_id": user_id,
                "track_id": track_id,
                "rating": row["rating"]
            })

    df_train = pd.DataFrame(train_rows)
    df_test = pd.DataFrame(test_rows)

    print(
        f"Training interactions: {len(df_train)}"
    )

    print(
        f"Testing interactions: {len(df_test)}"
    )

    # --------------------------------------------------------
    # 4. TRAIN FRESH SVD
    # --------------------------------------------------------

    print(
        "\nTraining fresh SVD..."
    )

    svd = TuneAISVD(
        n_components=50,
        random_state=RANDOM_STATE
    )

    svd.fit(df_train)

    print(
        "SVD training completed."
    )

    # --------------------------------------------------------
    # 5. LOAD RECOMMENDER
    # --------------------------------------------------------

    print(
        "\nLoading TuneAI engine..."
    )

    engine = TuneAIRecommender.load_engine()

    catalog = engine.df_songs.copy()

    # --------------------------------------------------------
    # 6. DEBUG EACH USER
    # --------------------------------------------------------

    for user_id in selected_users:

        print("\n")
        print("=" * 80)
        print(f"USER: {user_id}")
        print("=" * 80)

        train_tracks = set(
            df_train[
                df_train["user_id"] == user_id
            ]["track_id"]
            .astype(str)
        )

        test_tracks = set(
            df_test[
                df_test["user_id"] == user_id
            ]["track_id"]
            .astype(str)
        )

        if not train_tracks or not test_tracks:
            print("Skipping user.")
            continue

        # ----------------------------------------------------
        # TARGET
        # ----------------------------------------------------

        target_track = next(
            iter(train_tracks)
        )

        print(
            f"Target track: {target_track}"
        )

        print(
            f"Ground truth test tracks: "
            f"{len(test_tracks)}"
        )

        # ----------------------------------------------------
        # CANDIDATES
        # ----------------------------------------------------

        catalog_ids = (
            catalog["track_id"]
            .dropna()
            .astype(str)
            .unique()
        )

        if len(catalog_ids) > 2000:

            candidate_ids = set(
                rng.choice(
                    catalog_ids,
                    size=2000,
                    replace=False
                )
            )

        else:

            candidate_ids = set(
                catalog_ids
            )

        candidate_ids = list(
            (
                candidate_ids
                | test_tracks
            )
            - train_tracks
        )

        # ----------------------------------------------------
        # SVD SCORES
        # ----------------------------------------------------

        svd_scores = (
            svd.predict_user_scores_for_candidates(
                user_id,
                candidate_ids
            )
        )

        # ----------------------------------------------------
        # CONTENT SCORES
        # ----------------------------------------------------

        try:

            content_scores = (
                engine.similarity
                .calculate_pairwise_scores(
                    target_track,
                    candidate_ids,
                    catalog
                )
            )

        except Exception as e:

            print(
                f"Content scoring error: {e}"
            )

            content_scores = {}

        # ----------------------------------------------------
        # HYBRID SCORES
        # ----------------------------------------------------

        hybrid_scores = {}

        for track_id in candidate_ids:

            s = svd_scores.get(
                track_id,
                0.5
            )

            c = content_scores.get(
                track_id,
                0.5
            )

            hybrid_scores[track_id] = (
                SVD_WEIGHT * s
                +
                CONTENT_WEIGHT * c
            )

        # ----------------------------------------------------
        # SCORE STATISTICS
        # ----------------------------------------------------

        print("\nSCORE STATISTICS")
        print("-" * 80)

        if svd_scores:

            print(
                f"SVD     -> "
                f"min={min(svd_scores.values()):.4f}, "
                f"max={max(svd_scores.values()):.4f}, "
                f"mean={np.mean(list(svd_scores.values())):.4f}"
            )

        if content_scores:

            print(
                f"Content -> "
                f"min={min(content_scores.values()):.4f}, "
                f"max={max(content_scores.values()):.4f}, "
                f"mean={np.mean(list(content_scores.values())):.4f}"
            )

        if hybrid_scores:

            print(
                f"Hybrid  -> "
                f"min={min(hybrid_scores.values()):.4f}, "
                f"max={max(hybrid_scores.values()):.4f}, "
                f"mean={np.mean(list(hybrid_scores.values())):.4f}"
            )

        # ----------------------------------------------------
        # TOP SVD
        # ----------------------------------------------------

        print("\nTOP 10 SVD")
        print("-" * 80)

        top_svd = sorted(
            svd_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        for rank, (track_id, score) in enumerate(
            top_svd,
            start=1
        ):

            marker = (
                " <-- TEST"
                if track_id in test_tracks
                else ""
            )

            print(
                f"{rank:2}. "
                f"{track_id} "
                f"SVD={score:.4f}"
                f"{marker}"
            )

        # ----------------------------------------------------
        # TOP CONTENT
        # ----------------------------------------------------

        print("\nTOP 10 CONTENT")
        print("-" * 80)

        top_content = sorted(
            content_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        for rank, (track_id, score) in enumerate(
            top_content,
            start=1
        ):

            marker = (
                " <-- TEST"
                if track_id in test_tracks
                else ""
            )

            print(
                f"{rank:2}. "
                f"{track_id} "
                f"Content={score:.4f}"
                f"{marker}"
            )

        # ----------------------------------------------------
        # TOP HYBRID
        # ----------------------------------------------------

        print("\nTOP 10 HYBRID")
        print("-" * 80)

        top_hybrid = sorted(
            hybrid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        for rank, (track_id, score) in enumerate(
            top_hybrid,
            start=1
        ):

            s = svd_scores.get(
                track_id,
                0.5
            )

            c = content_scores.get(
                track_id,
                0.5
            )

            marker = (
                " <-- TEST"
                if track_id in test_tracks
                else ""
            )

            print(
                f"{rank:2}. "
                f"{track_id} "
                f"SVD={s:.4f} "
                f"Content={c:.4f} "
                f"Hybrid={score:.4f}"
                f"{marker}"
            )

    print("\n")
    print("=" * 80)
    print("DEBUG COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()