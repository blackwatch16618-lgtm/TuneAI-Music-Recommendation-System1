"""
TuneAI Content-Based Similarity Engine

Uses:
- StandardScaler for feature normalization
- Cosine Similarity for song-to-song similarity
- On-the-fly calculation to avoid large NxN similarity matrices
"""

from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from config import AUDIO_FEATURES, SCALER_PATH


class TuneAISimilarity:
    """
    Content-based music similarity engine.

    Songs are compared using audio features such as:
    danceability, energy, loudness, speechiness,
    acousticness, instrumentalness, liveness,
    valence, tempo and duration_ms.
    """

    def __init__(self, scaler: Optional[StandardScaler] = None):
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.is_fitted = False
        self.audio_features = AUDIO_FEATURES

    # ---------------------------------------------------------
    # FIT SCALER
    # ---------------------------------------------------------

    def fit(self, df_songs: pd.DataFrame) -> "TuneAISimilarity":
        """
        Fits StandardScaler using the song audio features.
        """

        missing_features = [
            feature
            for feature in self.audio_features
            if feature not in df_songs.columns
        ]

        if missing_features:
            raise KeyError(
                f"Missing required audio features: {missing_features}"
            )

        feature_matrix = (
            df_songs[self.audio_features]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .astype(np.float64)
            .values
        )

        self.scaler.fit(feature_matrix)
        self.is_fitted = True

        return self

    # ---------------------------------------------------------
    # TRANSFORM FEATURES
    # ---------------------------------------------------------

    def transform(self, df_songs: pd.DataFrame) -> np.ndarray:
        """
        Converts song audio features into standardized vectors.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Scaler is not fitted. Call fit() before transform()."
            )

        missing_features = [
            feature
            for feature in self.audio_features
            if feature not in df_songs.columns
        ]

        if missing_features:
            raise KeyError(
                f"Missing required audio features: {missing_features}"
            )

        feature_matrix = (
            df_songs[self.audio_features]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .astype(np.float64)
            .values
        )

        return self.scaler.transform(feature_matrix)

    # ---------------------------------------------------------
    # CALCULATE SIMILAR SONGS
    # ---------------------------------------------------------

    def calculate_similarities(
        self,
        target_track_id: str,
        df_songs: pd.DataFrame,
        candidate_ids=None,
        top_n: int = 10,
        year_filter: Optional[int] = None,
        genre_filter: Optional[str] = None
    ) -> List[Dict]:

        """
        Finds songs similar to the selected song.

        Returns the top N most similar songs.
        """

        if "track_id" not in df_songs.columns:
            raise KeyError(
                "The DataFrame must contain a 'track_id' column."
            )

        # Convert IDs to strings for consistent comparison
        working_df = df_songs.copy()
        working_df["track_id"] = working_df["track_id"].astype(str)

        target_track_id = str(target_track_id)

        # Check target song
        if target_track_id not in working_df["track_id"].values:
            raise ValueError(
                f"Target track '{target_track_id}' not found in catalog."
            )

        # Make sure model is fitted
        if not self.is_fitted:
            self.fit(working_df)

        # Get target song
        target_rows = working_df[
            working_df["track_id"] == target_track_id
        ]

        if target_rows.empty:
            raise ValueError(
                f"Target track '{target_track_id}' not found."
            )

        target_row = target_rows.iloc[0]

        # Target feature vector
        target_features = pd.DataFrame(
            [target_row[self.audio_features].values],
            columns=self.audio_features
        )

        target_vector = self.transform(target_features)

        # -----------------------------------------------------
        # BUILD CANDIDATE POOL
        # -----------------------------------------------------

        candidates_mask = (
            working_df["track_id"] != target_track_id
        )

        # Candidate ID filter
        if candidate_ids is not None:
            candidate_ids = [str(cid) for cid in candidate_ids]

            candidates_mask &= (
                working_df["track_id"].isin(candidate_ids)
            )

        # Year filter
        if (
            year_filter is not None
            and "release_year" in working_df.columns
        ):
            candidates_mask &= (
                pd.to_numeric(
                    working_df["release_year"],
                    errors="coerce"
                )
                == int(year_filter)
            )

        # Genre filter
        if (
            genre_filter
            and genre_filter.lower() != "all"
            and "genre" in working_df.columns
        ):
            candidates_mask &= (
                working_df["genre"]
                .fillna("")
                .astype(str)
                .str.lower()
                == genre_filter.lower()
            )

        candidates_df = working_df[candidates_mask].copy()

        if candidates_df.empty:
            return []

        # -----------------------------------------------------
        # CALCULATE COSINE SIMILARITY
        # -----------------------------------------------------

        candidate_vectors = self.transform(candidates_df)

        raw_similarities = cosine_similarity(
            target_vector,
            candidate_vectors
        )[0]

        # Convert [-1, 1] → [0, 1]
        normalized_similarities = (
            raw_similarities + 1.0
        ) / 2.0

        normalized_similarities = np.clip(
            normalized_similarities,
            0.0,
            1.0
        )

        candidates_df["similarity"] = (
            normalized_similarities
        )

        # Sort highest similarity first
        candidates_df = candidates_df.sort_values(
            by="similarity",
            ascending=False
        )

        top_results = candidates_df.head(
            max(1, int(top_n))
        )

        # -----------------------------------------------------
        # CREATE API RESPONSE
        # -----------------------------------------------------

        results = []

        for _, row in top_results.iterrows():

            similarity = float(row["similarity"])

            release_year = row.get(
                "release_year",
                0
            )

            try:
                release_year = int(
                    float(release_year)
                )
            except (ValueError, TypeError):
                release_year = 0

            popularity = row.get(
                "popularity",
                0
            )

            try:
                popularity = float(
                    popularity
                )
            except (ValueError, TypeError):
                popularity = 0.0

            preview_url = row.get(
                "preview_url",
                None
            )

            if pd.isna(preview_url):
                preview_url = None

            results.append(
                {
                    "track_id": str(
                        row["track_id"]
                    ),

                    "track_name": str(
                        row.get(
                            "track_name",
                            "Unknown Track"
                        )
                    ),

                    "artist": str(
                        row.get(
                            "artist",
                            "Unknown Artist"
                        )
                    ),

                    "album": str(
                        row.get(
                            "album_name",
                            ""
                        )
                    ),

                    "release_year": release_year,

                    "genre": str(
                        row.get(
                            "genre",
                            "Unknown"
                        )
                    ),

                    "similarity": round(
                        similarity,
                        4
                    ),

                    "similarity_display": (
                        f"{int(round(similarity * 100))}% Match"
                    ),

                    "popularity": round(
                        popularity,
                        1
                    ),

                    "preview_url": preview_url
                }
            )

        return results

    # ---------------------------------------------------------
    # PAIRWISE SIMILARITY
    # ---------------------------------------------------------

    def calculate_pairwise_scores(
        self,
        target_track_id: str,
        candidate_track_ids: List[str],
        df_songs: pd.DataFrame
    ) -> Dict[str, float]:

        """
        Calculates cosine similarity between one target song
        and a specific list of candidate songs.

        Returns:
            {
                "track_id": similarity_score
            }
        """

        working_df = df_songs.copy()

        working_df["track_id"] = (
            working_df["track_id"]
            .astype(str)
        )

        target_track_id = str(
            target_track_id
        )

        candidate_track_ids = [
            str(cid)
            for cid in candidate_track_ids
        ]

        # Target song does not exist
        if target_track_id not in working_df["track_id"].values:
            return {
                cid: 0.5
                for cid in candidate_track_ids
            }

        # Fit scaler if necessary
        if not self.is_fitted:
            self.fit(working_df)

        # Target song
        target_row = working_df[
            working_df["track_id"] == target_track_id
        ].iloc[0]

        target_features = pd.DataFrame(
            [target_row[self.audio_features].values],
            columns=self.audio_features
        )

        target_vector = self.transform(
            target_features
        )

        # Candidate songs
        candidates_df = working_df[
            working_df["track_id"].isin(
                candidate_track_ids
            )
        ].copy()

        if candidates_df.empty:
            return {}

        candidate_vectors = self.transform(
            candidates_df
        )

        # Calculate cosine similarity
        raw_similarities = cosine_similarity(
            target_vector,
            candidate_vectors
        )[0]

        # Convert [-1, 1] to [0, 1]
        normalized_similarities = (
            raw_similarities + 1.0
        ) / 2.0

        normalized_similarities = np.clip(
            normalized_similarities,
            0.0,
            1.0
        )

        return {
            str(track_id): float(similarity)
            for track_id, similarity
            in zip(
                candidates_df["track_id"],
                normalized_similarities
            )
        }

    # ---------------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------------

    def save(
        self,
        scaler_path: Path = SCALER_PATH
    ):
        """
        Saves the fitted StandardScaler using joblib.
        """

        scaler_path = Path(
            scaler_path
        )

        scaler_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            {
                "scaler": self.scaler,
                "is_fitted": self.is_fitted
            },
            scaler_path
        )

    # ---------------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------------

    @classmethod
    def load(
        cls,
        scaler_path: Path = SCALER_PATH
    ) -> "TuneAISimilarity":

        """
        Loads a previously saved scaler.
        """

        scaler_path = Path(
            scaler_path
        )

        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler not found at: {scaler_path}"
            )

        data = joblib.load(
            scaler_path
        )

        instance = cls(
            scaler=data["scaler"]
        )

        instance.is_fitted = data.get(
            "is_fitted",
            True
        )

        return instance