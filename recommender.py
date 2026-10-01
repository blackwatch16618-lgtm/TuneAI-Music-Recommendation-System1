"""
TuneAI Hybrid Recommendation Engine
Orchestrates collaborative filtering (SVD) and content-based filtering (Cosine Similarity)
to generate personalized, high-precision music recommendations.
Formula:
    final_score = SVD_WEIGHT * SVD_score + CONTENT_WEIGHT * content_similarity
Includes automatic fallbacks for cold-start users and songs outside the latent model.
"""

from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from config import (
    SVD_WEIGHT, CONTENT_WEIGHT, DEFAULT_LIMIT,
    SVD_MODEL_PATH, SCALER_PATH, PROCESSED_DATA_PATH
)
from svd_model import TuneAISVD
from similarity import TuneAISimilarity
from database import Song, ListeningHistory, SessionLocal


class TuneAIRecommender:
    """
    Core recommendation engine combining:
      1. Collaborative Filtering (TruncatedSVD)
      2. Content-Based Cosine Similarity (Normalized audio features)
      3. User listening history filtering
      4. Intelligent cold-start fallback
    """

    def __init__(self, svd_model: Optional[TuneAISVD] = None, similarity: Optional[TuneAISimilarity] = None, df_songs: Optional[pd.DataFrame] = None):
        self.svd_model = svd_model
        self.similarity = similarity
        self.df_songs = df_songs if df_songs is not None else pd.DataFrame()
        self.svd_weight = SVD_WEIGHT
        self.content_weight = CONTENT_WEIGHT
        self.is_ready = False

    @classmethod
    def load_engine(cls) -> "TuneAIRecommender":
        """Factory method to load trained models and catalog into memory at application start."""
        instance = cls()
        print("[TuneAI] Initializing Recommendation Engine...")

        # Load Processed Songs Catalog
        if PROCESSED_DATA_PATH.exists():
            instance.df_songs = pd.read_csv(
                PROCESSED_DATA_PATH,
                low_memory=False
            )

            # Clean release year values
            instance.df_songs["release_year"] = pd.to_numeric(
                instance.df_songs["release_year"],
                errors="coerce"
            ).fillna(0).astype(int)

            print(f"[TuneAI] Catalog loaded: {len(instance.df_songs):,} songs.")
            
        else:
            print("[TuneAI] Warning: Processed songs CSV not found. Loading from database...")
            db = SessionLocal()
            try:
                songs = db.query(Song).all()
                if songs:
                    instance.df_songs = pd.DataFrame([s.to_dict() for s in songs])
            finally:
                db.close()

        # Load SVD Model
        if SVD_MODEL_PATH.exists():
            instance.svd_model = TuneAISVD.load()
            print(f"[TuneAI] SVD model loaded with {instance.svd_model.n_components} components.")
        else:
            print("[TuneAI] SVD model not found on disk. Engine will run in content-only fallback mode until trained.")
            instance.svd_model = TuneAISVD()

        # Load Scaler
        if SCALER_PATH.exists():
            instance.similarity = TuneAISimilarity.load()
            print("[TuneAI] Scaler and content similarity engine loaded.")
        else:
            print("[TuneAI] Scaler not found on disk. Initializing new similarity engine...")
            instance.similarity = TuneAISimilarity()
            if len(instance.df_songs) > 0:
                instance.similarity.fit(instance.df_songs)

        instance.is_ready = True
        return instance

    def get_user_heard_tracks(self, user_id: str, db: Optional[Session] = None) -> Set[str]:
        """Retrieves the set of track IDs already heard by a specific user."""
        if not user_id:
            return set()

        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True

        try:
            records = db.query(ListeningHistory.track_id).filter(ListeningHistory.user_id == user_id).all()
            return {r[0] for r in records}
        finally:
            if close_session:
                db.close()

    def get_track_by_id(self, track_id: str) -> Optional[Dict]:
        """Returns full metadata dictionary for a single track ID."""
        if len(self.df_songs) == 0:
            return None
        match = self.df_songs[self.df_songs["track_id"] == track_id]
        if len(match) == 0:
            return None
        row = match.iloc[0]
        return {
            "track_id": str(row["track_id"]),
            "track_name": str(row["track_name"]),
            "artist": str(row["artist"]),
            "album": str(row.get("album_name", "Single / Unknown Album")),
            "release_date": str(row.get("release_date", row["release_year"])),
            "release_year": int(row["release_year"]),
            "genre": str(row.get("genre", "Pop")),
            "popularity": round(float(row.get("popularity", 0)), 1),
            "duration_ms": float(row.get("duration_ms", 0)),
            "danceability": round(float(row.get("danceability", 0)), 3),
            "energy": round(float(row.get("energy", 0)), 3),
            "loudness": round(float(row.get("loudness", 0)), 2),
            "speechiness": round(float(row.get("speechiness", 0)), 3),
            "acousticness": round(float(row.get("acousticness", 0)), 3),
            "instrumentalness": round(float(row.get("instrumentalness", 0)), 4),
            "liveness": round(float(row.get("liveness", 0)), 3),
            "valence": round(float(row.get("valence", 0)), 3),
            "tempo": round(float(row.get("tempo", 0)), 1),
            "explicit": bool(row.get("explicit", False)),
            "preview_url": row.get("preview_url") if pd.notna(row.get("preview_url")) else None
        }

    def search_songs(
        self,
        query: str,
        limit: int = 20,
        year_filter: Optional[int] = None,
        genre_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Fast, case-insensitive partial-match search across track_name, artist, and album_name.
        """
        if len(self.df_songs) == 0:
            return []

        q = (query or "").strip().lower()
        if not q:
            # Return top popular songs if query is empty
            return self.get_popular_songs(limit=limit, year_filter=year_filter, genre_filter=genre_filter)

        mask = (
            self.df_songs["track_name"].astype(str).str.lower().str.contains(q, na=False) |
            self.df_songs["artist"].astype(str).str.lower().str.contains(q, na=False) |
            self.df_songs["album_name"].astype(str).str.lower().str.contains(q, na=False)
        )

        if year_filter is not None:
            mask &= (self.df_songs["release_year"] == int(year_filter))
        if genre_filter and genre_filter.lower() != "all":
            mask &= (self.df_songs["genre"].str.lower() == genre_filter.lower())

        results_df = self.df_songs[mask].sort_values(by="popularity", ascending=False).head(limit)

        results = []
        for _, row in results_df.iterrows():
            results.append({
                "track_id": str(row["track_id"]),
                "track_name": str(row["track_name"]),
                "artist": str(row["artist"]),
                "album": str(row.get("album_name", "")),
                "release_year": int(row["release_year"]),
                "genre": str(row.get("genre", "Pop")),
                "popularity": round(float(row.get("popularity", 0)), 1),
                "preview_url": row.get("preview_url") if pd.notna(row.get("preview_url")) else None
            })

        return results

    def get_popular_songs(
        self,
        limit: int = 20,
        year_filter: Optional[int] = None,
        genre_filter: Optional[str] = None
    ) -> List[Dict]:
        """Returns top songs ranked by popularity score."""
        if len(self.df_songs) == 0:
            return []

        df = self.df_songs.copy()
        if year_filter is not None:
            df = df[df["release_year"] == int(year_filter)]
        if genre_filter and genre_filter.lower() != "all":
            df = df[df["genre"].str.lower() == genre_filter.lower()]

        df = df.sort_values(by="popularity", ascending=False).head(limit)

        results = []
        for _, row in df.iterrows():
            results.append({
                "track_id": str(row["track_id"]),
                "track_name": str(row["track_name"]),
                "artist": str(row["artist"]),
                "album": str(row.get("album_name", "")),
                "release_year": int(row["release_year"]),
                "genre": str(row.get("genre", "Pop")),
                "popularity": round(float(row.get("popularity", 0)), 1),
                "preview_url": row.get("preview_url") if pd.notna(row.get("preview_url")) else None
            })
        return results

    def recommend_similar(
        self,
        track_id: str,
        limit: int = DEFAULT_LIMIT,
        year_filter: Optional[int] = None,
        genre_filter: Optional[str] = None
    ) -> List[Dict]:
        """Pure content-based recommendations using audio feature cosine similarity."""
        if not self.similarity or not self.similarity.is_fitted:
            if len(self.df_songs) > 0:
                self.similarity = TuneAISimilarity().fit(self.df_songs)
            else:
                return []

        return self.similarity.calculate_similarities(
            target_track_id=track_id,
            df_songs=self.df_songs,
            top_n=limit,
            year_filter=year_filter,
            genre_filter=genre_filter
        )

    def recommend_hybrid(
        self,
        track_id: str,
        user_id: Optional[str] = None,
        limit: int = DEFAULT_LIMIT,
        year_filter: Optional[int] = None,
        genre_filter: Optional[str] = None
    ) -> Dict:
        """
        Executes the complete TuneAI hybrid recommendation pipeline:
          1. Select Song & Validate Target
          2. Check User History
          3. Candidate Generation
          4. Collaborative Filtering (SVD) Scoring
          5. Content-Based Cosine Similarity Scoring
          6. Hybrid Score Combination (0.70 SVD + 0.30 Content)
          7. Filter Heard Tracks & Selected Track
          8. Rank and Return Top N
        """
        target_song = self.get_track_by_id(track_id)
        if not target_song:
            raise ValueError(f"Track with ID '{track_id}' not found in TuneAI catalog.")

        heard_track_ids: Set[str] = set()
        user_has_history = False
        if user_id:
            heard_track_ids = self.get_user_heard_tracks(user_id)
            if user_id in getattr(self.svd_model, "user_to_idx", {}) or len(heard_track_ids) > 0:
                user_has_history = True

        # Fallback condition 1: No user history or SVD model not ready -> use pure content similarity
        svd_available = (
            self.svd_model is not None and
            self.svd_model.is_fitted and
            user_has_history and
            user_id in self.svd_model.user_to_idx
        )

        if not svd_available:
            content_results = self.recommend_similar(
                track_id=track_id,
                limit=limit,
                year_filter=year_filter,
                genre_filter=genre_filter
            )
            # Filter out heard songs if any
            filtered = [r for r in content_results if r["track_id"] not in heard_track_ids]
            final_recs = filtered[:limit] if filtered else content_results[:limit]
            for r in final_recs:
                r["recommendation_score"] = r["similarity"]
            return {
                "engine": "TuneAI Content-Based Fallback Engine",
                "mode": "content_only",
                "track": target_song,
                "user_id": user_id,
                "recommendations": final_recs
            }

        # Step 3: Candidate Generation
        # Filter candidate pool
        mask = (self.df_songs["track_id"] != track_id)
        if heard_track_ids:
            mask &= (~self.df_songs["track_id"].isin(heard_track_ids))
        if year_filter is not None:
            mask &= (self.df_songs["release_year"] == int(year_filter))
        if genre_filter and genre_filter.lower() != "all":
            mask &= (self.df_songs["genre"].str.lower() == genre_filter.lower())

        candidates_df = self.df_songs[mask]
        if len(candidates_df) == 0:
            return {
                "engine": "TuneAI Hybrid Recommendation Engine",
                "mode": "hybrid",
                "track": target_song,
                "user_id": user_id,
                "recommendations": []
            }

        # Limit candidate evaluation to top candidates by popularity/diversity to keep it snappy
        if len(candidates_df) > 500:
            candidates_df = candidates_df.sort_values(by="popularity", ascending=False).head(500)

        candidate_ids = candidates_df["track_id"].tolist()

        # Step 4: SVD Collaborative Filtering Scores
        svd_scores = self.svd_model.predict_user_scores_for_candidates(user_id, candidate_ids)

        # Step 5: Content-Based Cosine Similarity Scores
        content_scores = self.similarity.calculate_pairwise_scores(track_id, candidate_ids, self.df_songs)

        # Step 6: Hybrid Scoring
        scored_candidates = []
        for _, row in candidates_df.iterrows():
            cid = str(row["track_id"])
            s_score = svd_scores.get(cid, 0.5)
            c_score = content_scores.get(cid, 0.5)

            final_score = (self.svd_weight * s_score) + (self.content_weight * c_score)
            scored_candidates.append({
                "track_id": cid,
                "track_name": str(row["track_name"]),
                "artist": str(row["artist"]),
                "album": str(row.get("album_name", "")),
                "release_year": int(row["release_year"]),
                "genre": str(row.get("genre", "Pop")),
                "similarity": round(c_score, 4),
                "similarity_display": f"{int(round(c_score * 100))}% Match",
                "recommendation_score": round(final_score, 4),
                "popularity": round(float(row.get("popularity", 0)), 1),
                "preview_url": row.get("preview_url") if pd.notna(row.get("preview_url")) else None
            })

        # Step 7 & 8: Sort and Return Top N
        scored_candidates.sort(key=lambda x: x["recommendation_score"], reverse=True)
        final_recs = scored_candidates[:limit]

        return {
            "engine": "TuneAI Hybrid Recommendation Engine",
            "mode": "hybrid",
            "track": target_song,
            "user_id": user_id,
            "recommendations": final_recs
        }
