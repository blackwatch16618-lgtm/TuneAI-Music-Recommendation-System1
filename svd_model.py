"""
TuneAI SVD Matrix Factorization Model
Implements collaborative filtering using TruncatedSVD over sparse user-song
interaction matrices. Learns latent representations for users and songs,
predicts preference scores, and saves/loads model parameters via joblib.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from config import (
    N_COMPONENTS, RANDOM_STATE, MODELS_DIR,
    SVD_MODEL_PATH, SONG_INDEX_PATH, USER_INDEX_PATH
)


class TuneAISVD:
    """
    Collaborative filtering model powered by TruncatedSVD matrix factorization.
    Learns dense latent embeddings for users and songs from sparse interaction matrices.
    """

    def __init__(self, n_components: int = N_COMPONENTS, random_state: int = RANDOM_STATE):
        self.n_components = n_components
        self.random_state = random_state
        self.svd: Optional[TruncatedSVD] = None
        self.user_factors: Optional[np.ndarray] = None
        self.song_factors: Optional[np.ndarray] = None
        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.song_to_idx: Dict[str, int] = {}
        self.idx_to_song: Dict[int, str] = {}
        self.min_score: float = 0.0
        self.max_score: float = 1.0
        self.is_fitted: bool = False

    def build_sparse_matrix(self, df_interactions: pd.DataFrame) -> Tuple[csr_matrix, Dict[str, int], Dict[str, int]]:
        """
        Constructs a memory-efficient sparse CSR matrix from interaction triples.
        Required columns: user_id, track_id, rating
        """
        unique_users = df_interactions["user_id"].unique()
        unique_songs = df_interactions["track_id"].unique()

        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.song_to_idx = {s: i for i, s in enumerate(unique_songs)}
        self.idx_to_song = {i: s for s, i in self.song_to_idx.items()}

        row_indices = df_interactions["user_id"].map(self.user_to_idx).values
        col_indices = df_interactions["track_id"].map(self.song_to_idx).values
        ratings = df_interactions["rating"].values.astype(np.float32)

        n_users = len(self.user_to_idx)
        n_songs = len(self.song_to_idx)

        sparse_matrix = csr_matrix((ratings, (row_indices, col_indices)), shape=(n_users, n_songs), dtype=np.float32)
        return sparse_matrix, self.user_to_idx, self.song_to_idx

    def fit(self, df_interactions: pd.DataFrame) -> "TuneAISVD":
        """Fits TruncatedSVD on the interaction records and computes latent factors."""
        if len(df_interactions) == 0:
            raise ValueError("Interactions DataFrame is empty.")

        sparse_matrix, _, _ = self.build_sparse_matrix(df_interactions)

        # If available songs or users are fewer than requested n_components, adjust safely
        max_possible_components = min(sparse_matrix.shape[0] - 1, sparse_matrix.shape[1] - 1, self.n_components)
        actual_components = max(2, max_possible_components)

        self.svd = TruncatedSVD(
            n_components=actual_components,
            algorithm="randomized",
            random_state=self.random_state
        )

        # user_factors: (n_users, n_components)
        self.user_factors = self.svd.fit_transform(sparse_matrix)
        # song_factors: (n_songs, n_components)
        self.song_factors = self.svd.components_.T

        # Normalize score range for scaling predictions to [0, 1]
        sample_scores = np.sum(self.user_factors[:100] @ self.song_factors[:100].T, axis=1)
        self.min_score = float(np.min(sample_scores)) if len(sample_scores) > 0 else 0.0
        self.max_score = float(np.max(sample_scores)) if len(sample_scores) > 0 else 1.0
        if self.max_score <= self.min_score:
            self.max_score = self.min_score + 1.0

        self.is_fitted = True
        return self

    def predict(self, user_id: str, track_id: str) -> float:
        """
        Predicts a preference score [0, 1] for a specific user and song.
        Returns 0.0 if user or track is unseen.
        """
        if not self.is_fitted or user_id not in self.user_to_idx or track_id not in self.song_to_idx:
            return 0.0

        u_idx = self.user_to_idx[user_id]
        s_idx = self.song_to_idx[track_id]

        raw_score = float(np.dot(self.user_factors[u_idx], self.song_factors[s_idx]))
        norm_score = (raw_score - self.min_score) / (self.max_score - self.min_score)
        return float(np.clip(norm_score, 0.0, 1.0))

    def predict_user_scores_for_candidates(self, user_id: str, candidate_track_ids: List[str]) -> Dict[str, float]:
        """
        Vectorized prediction for a given user across multiple candidate track IDs.
        Returns a dictionary mapping track_id to normalized SVD score [0, 1].
        """
        scores: Dict[str, float] = {}
        if not self.is_fitted or user_id not in self.user_to_idx:
            return {tid: 0.5 for tid in candidate_track_ids}

        u_idx = self.user_to_idx[user_id]
        u_vector = self.user_factors[u_idx]

        valid_indices = []
        valid_tids = []
        for tid in candidate_track_ids:
            if tid in self.song_to_idx:
                valid_indices.append(self.song_to_idx[tid])
                valid_tids.append(tid)
            else:
                scores[tid] = 0.5

        if valid_indices:
            cand_factors = self.song_factors[valid_indices]
            raw_scores = u_vector @ cand_factors.T
            norm_scores = (raw_scores - self.min_score) / (self.max_score - self.min_score)
            norm_scores = np.clip(norm_scores, 0.0, 1.0)
            for tid, sc in zip(valid_tids, norm_scores):
                scores[tid] = float(sc)

        return scores

    def save(self, models_dir: Path = MODELS_DIR):
        """Saves model parameters and index mappings via joblib."""
        models_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_components": self.n_components,
            "svd": self.svd,
            "user_factors": self.user_factors,
            "song_factors": self.song_factors,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "is_fitted": self.is_fitted
        }
        joblib.dump(payload, SVD_MODEL_PATH)
        joblib.dump({"user_to_idx": self.user_to_idx, "idx_to_user": self.idx_to_user}, USER_INDEX_PATH)
        joblib.dump({"song_to_idx": self.song_to_idx, "idx_to_song": self.idx_to_song}, SONG_INDEX_PATH)

    @classmethod
    def load(cls, models_dir: Path = MODELS_DIR) -> "TuneAISVD":
        """Loads fitted model and indices from disk."""
        if not SVD_MODEL_PATH.exists() or not USER_INDEX_PATH.exists() or not SONG_INDEX_PATH.exists():
            raise FileNotFoundError("SVD model or index files not found in models directory.")

        instance = cls()
        payload = joblib.load(SVD_MODEL_PATH)
        instance.n_components = payload["n_components"]
        instance.svd = payload["svd"]
        instance.user_factors = payload["user_factors"]
        instance.song_factors = payload["song_factors"]
        instance.min_score = payload.get("min_score", 0.0)
        instance.max_score = payload.get("max_score", 1.0)
        instance.is_fitted = payload["is_fitted"]

        user_data = joblib.load(USER_INDEX_PATH)
        instance.user_to_idx = user_data["user_to_idx"]
        instance.idx_to_user = user_data["idx_to_user"]

        song_data = joblib.load(SONG_INDEX_PATH)
        instance.song_to_idx = song_data["song_to_idx"]
        instance.idx_to_song = song_data["idx_to_song"]

        return instance
