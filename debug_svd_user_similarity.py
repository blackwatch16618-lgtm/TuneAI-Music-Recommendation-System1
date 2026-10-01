import pandas as pd
import numpy as np

from config import RANDOM_STATE, PROCESSED_DATA_PATH
from database import SessionLocal, ListeningHistory
from svd_model import TuneAISVD


USER_ID = "user_11734"


def main():

    print("=" * 80)
    print("        TuneAI USER / HELD-OUT SONG SIMILARITY CHECK")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Load interactions
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

    # --------------------------------------------------------
    # 2. Same 80/20 split
    # --------------------------------------------------------

    train_rows = []
    test_rows = []

    for user_id, user_df in df.groupby("user_id"):

        tracks = list(
            dict.fromkeys(
                user_df["track_id"]
                .astype(str)
                .tolist()
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

    user_train = df_train[
        df_train["user_id"] == USER_ID
    ]

    user_test = df_test[
        df_test["user_id"] == USER_ID
    ]

    train_tracks = user_train[
        "track_id"
    ].astype(str).tolist()

    test_tracks = user_test[
        "track_id"
    ].astype(str).tolist()

    print(f"\nUser: {USER_ID}")

    print(
        f"Training tracks: {len(train_tracks)}"
    )

    print(
        f"Held-out tracks: {len(test_tracks)}"
    )

    print("\nTRAINING TRACKS:")

    for track in train_tracks:
        print(f"  {track}")

    print("\nHELD-OUT TRACKS:")

    for track in test_tracks:
        print(f"  {track}")

    # --------------------------------------------------------
    # 3. Train SVD
    # --------------------------------------------------------

    print("\nTraining SVD...")

    svd = TuneAISVD(
        n_components=50,
        random_state=RANDOM_STATE
    )

    svd.fit(df_train)

    # --------------------------------------------------------
    # 4. Compare latent vectors
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "LATENT VECTOR SIMILARITY"
    )

    print(
        "=" * 80
    )

    results = []

    for test_track in test_tracks:

        if test_track not in svd.song_to_idx:
            print(
                f"\n{test_track}: "
                "NOT IN SVD"
            )
            continue

        test_idx = svd.song_to_idx[
            test_track
        ]

        test_vector = (
            svd.song_factors[test_idx]
        )

        similarities = []

        for train_track in train_tracks:

            if train_track not in svd.song_to_idx:
                continue

            train_idx = svd.song_to_idx[
                train_track
            ]

            train_vector = (
                svd.song_factors[train_idx]
            )

            denominator = (
                np.linalg.norm(test_vector)
                * np.linalg.norm(train_vector)
            )

            if denominator == 0:
                similarity = 0.0
            else:
                similarity = (
                    np.dot(
                        test_vector,
                        train_vector
                    )
                    / denominator
                )

            similarities.append(
                similarity
            )

        if similarities:

            max_similarity = max(
                similarities
            )

            avg_similarity = np.mean(
                similarities
            )

            print(
                f"\nHeld-out: {test_track}"
            )

            print(
                f"  Max similarity to "
                f"training songs: "
                f"{max_similarity:.4f}"
            )

            print(
                f"  Average similarity: "
                f"{avg_similarity:.4f}"
            )

            results.append({
                "track_id": test_track,
                "max_similarity":
                    max_similarity,
                "avg_similarity":
                    avg_similarity
            })

    # --------------------------------------------------------
    # 5. Summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "INTERPRETATION DATA"
    )

    print(
        "=" * 80
    )

    if results:

        result_df = pd.DataFrame(
            results
        )

        print(
            "\nAverage MAX similarity:"
        )

        print(
            f"{result_df['max_similarity'].mean():.4f}"
        )

        print(
            "\nAverage similarity:"
        )

        print(
            f"{result_df['avg_similarity'].mean():.4f}"
        )

    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()