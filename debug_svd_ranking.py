import numpy as np
import pandas as pd

from config import RANDOM_STATE
from database import SessionLocal, ListeningHistory
from svd_model import TuneAISVD


def main():

    print("=" * 80)
    print("              TuneAI SVD RANKING SANITY CHECK")
    print("=" * 80)

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print("\n[1/6] Loading interactions...")

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
    # 2. CREATE SAME 80/20 SPLIT
    # ========================================================

    print("\n[2/6] Creating 80/20 split...")

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

    df_train = pd.DataFrame(
        train_rows
    )

    df_test = pd.DataFrame(
        test_rows
    )

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
    # 3. TRAIN SVD
    # ========================================================

    print("\n[3/6] Training fresh SVD...")

    svd = TuneAISVD(
        n_components=50,
        random_state=RANDOM_STATE
    )

    svd.fit(df_train)

    print(
        f"SVD users: {len(svd.user_to_idx):,}"
    )

    print(
        f"SVD songs: {len(svd.song_to_idx):,}"
    )

    # ========================================================
    # 4. SELECT ONE USER
    # ========================================================

    print("\n[4/6] Selecting test user...")

    # Use users that exist in both train and test.
    train_users = set(
        df_train["user_id"]
    )

    test_users = set(
        df_test["user_id"]
    )

    possible_users = list(
        train_users & test_users
    )

    user_id = possible_users[
        0
    ]

    print(
        f"Selected user: {user_id}"
    )

    # ========================================================
    # 5. GET TRAIN / TEST TRACKS
    # ========================================================

    train_tracks = list(
        dict.fromkeys(
            df_train[
                df_train["user_id"] == user_id
            ]["track_id"]
            .astype(str)
            .tolist()
        )
    )

    test_tracks = list(
        dict.fromkeys(
            df_test[
                df_test["user_id"] == user_id
            ]["track_id"]
            .astype(str)
            .tolist()
        )
    )

    print(
        f"\nTraining tracks: "
        f"{len(train_tracks)}"
    )

    print(
        f"Test tracks: "
        f"{len(test_tracks)}"
    )

    print("\nHELD-OUT TEST TRACKS:")

    for track_id in test_tracks:

        known = (
            track_id
            in svd.song_to_idx
        )

        print(
            f"  {track_id} "
            f"-> SVD known: {known}"
        )

    # ========================================================
    # 6. RANK ALL KNOWN SVD SONGS
    # ========================================================

    print(
        "\n[5/6] Generating SVD scores..."
    )

    # IMPORTANT:
    # Only use songs known by the fresh SVD.

    known_songs = list(
        svd.song_to_idx.keys()
    )

    # Remove songs already heard by user.
    candidates = [
        track_id
        for track_id in known_songs
        if track_id not in train_tracks
    ]

    print(
        f"SVD-known candidates: "
        f"{len(candidates):,}"
    )

    scores = (
        svd.predict_user_scores_for_candidates(
            user_id,
            candidates
        )
    )

    print(
        f"Scores generated: "
        f"{len(scores):,}"
    )

    # ========================================================
    # SCORE DISTRIBUTION
    # ========================================================

    values = np.array(
        list(scores.values()),
        dtype=float
    )

    print("\nSVD SCORE DISTRIBUTION:")

    print(
        f"Minimum: "
        f"{values.min():.6f}"
    )

    print(
        f"Maximum: "
        f"{values.max():.6f}"
    )

    print(
        f"Mean: "
        f"{values.mean():.6f}"
    )

    print(
        f"Median: "
        f"{np.median(values):.6f}"
    )

    # ========================================================
    # RANK TEST TRACKS
    # ========================================================

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    rank_map = {
        track_id: rank
        for rank, (track_id, score)
        in enumerate(
            ranked,
            start=1
        )
    }

    print(
        "\nHELD-OUT TRACK RANKS:"
    )

    for track_id in test_tracks:

        if track_id in scores:

            rank = rank_map[
                track_id
            ]

            score = scores[
                track_id
            ]

            print(
                f"  {track_id}"
            )

            print(
                f"    Score: "
                f"{score:.6f}"
            )

            print(
                f"    Rank: "
                f"{rank:,} / {len(ranked):,}"
            )

            print(
                f"    In Top-5: "
                f"{rank <= 5}"
            )

            print(
                f"    In Top-10: "
                f"{rank <= 10}"
            )

        else:

            print(
                f"  {track_id}"
            )

            print(
                "    NOT AVAILABLE "
                "IN SVD SCORES"
            )

    # ========================================================
    # TOP 10
    # ========================================================

    print(
        "\nTOP 10 SVD RECOMMENDATIONS:"
    )

    for rank, (
        track_id,
        score
    ) in enumerate(
        ranked[:10],
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"{track_id} "
            f"score={score:.6f}"
        )

    print(
        "\n[6/6] SVD sanity check completed."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()