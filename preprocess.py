"""
TuneAI Data Preprocessing Pipeline
Loads raw CSV data, maps flexible column aliases, parses release dates,
filters strictly to 2022-2024, eliminates duplicates and nulls, cleans audio features,
saves the processed dataset to data/processed/tuneai_songs_2022_2024.csv,
and populates the SQLite database.
"""

import sys
import re
import pandas as pd
import numpy as np
from pathlib import Path
from config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, DB_PATH,
    COLUMN_MAPPINGS, AUDIO_FEATURES, ALLOWED_YEARS
)
from database import init_db, Song, SessionLocal


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Maps alternative column names in the raw dataset to standard TuneAI names."""
    df_mapped = df.copy()
    existing_cols = {col.lower().strip(): col for col in df.columns}

    rename_dict = {}
    for target_col, aliases in COLUMN_MAPPINGS.items():
        if target_col in df.columns:
            continue
        for alias in aliases:
            alias_lower = alias.lower().strip()
            if alias_lower in existing_cols:
                rename_dict[existing_cols[alias_lower]] = target_col
                break

    if rename_dict:
        df_mapped = df_mapped.rename(columns=rename_dict)

    return df_mapped


def clean_genre(val) -> str:
    """Normalizes raw genre strings into clean, readable genre tags."""
    if pd.isna(val) or val is None:
        return "Pop"
    s = str(val).strip()
    if not s or s in ("[]", "nan", "None", "unknown", "none"):
        return "Pop"

    # Handle python list string representation e.g. "['pop', 'dance pop']"
    if s.startswith("[") and s.endswith("]"):
        items = re.findall(r"'([^']+)'|\"([^\"]+)\"", s)
        flattened = [i[0] or i[1] for i in items if (i[0] or i[1])]
        if flattened:
            s = flattened[0]
        else:
            return "Pop"

    # Clean punctuation and capitalize
    s = s.replace("-", " ").title()
    return s


def preprocess_data(raw_csv_path: Path = RAW_DATA_PATH, output_csv_path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Executes the full preprocessing pipeline on the raw music dataset."""
    print("=" * 60)
    print("             TuneAI DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {raw_csv_path}. Please place raw data first.")

    print(f"[1/8] Loading raw dataset: {raw_csv_path}")
    try:
        df = pd.read_csv(raw_csv_path, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(raw_csv_path, low_memory=False, encoding="latin1")

    initial_count = len(df)
    print(f"      Raw rows loaded: {initial_count:,}")

    # Step 2: Apply Column Mapping Layer
    print("[2/8] Applying column mapping layer...")
    df = map_columns(df)

    # Validate essential columns
    if "track_name" not in df.columns:
        raise KeyError("Could not find a valid track name column using column mappings.")
    if "artist" not in df.columns:
        raise KeyError("Could not find a valid artist column using column mappings.")

    # Step 3: Remove records missing core metadata
    print("[3/8] Removing rows with missing track names or artists...")
    df = df.dropna(subset=["track_name", "artist"])
    df = df[df["track_name"].astype(str).str.strip() != ""]
    df = df[df["artist"].astype(str).str.strip() != ""]
    print(f"      Rows after non-empty metadata filter: {len(df):,}")

    # Step 4: Detect and parse release date, extract release year
    print("[4/8] Detecting release dates and extracting release years...")
    if "release_date" in df.columns:
        # Convert to datetime handling various formats
        parsed_dates = pd.to_datetime(df["release_date"], errors="coerce")
        if "release_year" not in df.columns or df["release_year"].isna().all():
            df["release_year"] = parsed_dates.dt.year
        else:
            df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
            df["release_year"] = df["release_year"].fillna(parsed_dates.dt.year)
    elif "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    else:
        raise KeyError("Neither release_date nor release_year column could be detected.")

    # Step 5: Filter strictly for 2022 <= release_year <= 2024
    print("[5/8] Filtering dataset to 2022 <= release_year <= 2024...")
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    df = df.dropna(subset=["release_year"])
    df["release_year"] = df["release_year"].astype(int)

    year_counts_before = df["release_year"].value_counts().to_dict()
    df = df[df["release_year"].isin(ALLOWED_YEARS)]
    print(f"      Rows strictly within 2022-2024: {len(df):,}")
    for y in sorted(ALLOWED_YEARS):
        print(f"        • Year {y}: {df[df['release_year'] == y].shape[0]:,} songs")

    # Step 6: Deduplication
    print("[6/8] Removing duplicate tracks...")
    if "track_id" in df.columns and not df["track_id"].isna().all():
        df = df.drop_duplicates(subset=["track_id"])
    df = df.drop_duplicates(subset=["track_name", "artist"])
    print(f"      Rows after deduplication: {len(df):,}")

    # Step 7: Clean Numerical Features and Audio Attributes
    print("[7/8] Cleaning numerical audio features and popularity...")
    if "popularity" not in df.columns:
        df["popularity"] = 50.0
    else:
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(50.0)
        df["popularity"] = df["popularity"].clip(0.0, 100.0)

    for feature in AUDIO_FEATURES:
        if feature in df.columns:
            df[feature] = pd.to_numeric(df[feature], errors="coerce")
            median_val = df[feature].median()
            df[feature] = df[feature].fillna(median_val if not pd.isna(median_val) else 0.5)
        else:
            # Default sensible values if a specific feature column was missing
            default_val = 180000.0 if feature == "duration_ms" else (120.0 if feature == "tempo" else (-8.0 if feature == "loudness" else 0.5))
            df[feature] = default_val

    # Clean genre
    if "genre" in df.columns:
        df["genre"] = df["genre"].apply(clean_genre)
    else:
        df["genre"] = "Pop"

    # Ensure album_name exists
    if "album_name" not in df.columns:
        df["album_name"] = df["track_name"] + " - Single"
    else:
        df["album_name"] = df["album_name"].fillna(df["track_name"] + " - Single")

    # Ensure track_id exists
    if "track_id" not in df.columns or df["track_id"].isna().any():
        import hashlib
        def gen_track_id(row):
            s = f"{row['track_name']}_{row['artist']}_{row['release_year']}"
            return "t_" + hashlib.md5(s.encode("utf-8")).hexdigest()[:16]
        df["track_id"] = [gen_track_id(r) for _, r in df.iterrows()]

    if "explicit" not in df.columns:
        df["explicit"] = False
    else:
        df["explicit"] = df["explicit"].astype(bool)

    if "preview_url" not in df.columns:
        df["preview_url"] = None

    # Step 8: Save Processed Dataset and Populate SQLite Database
    print(f"[8/8] Saving processed dataset to: {output_csv_path}")
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding="utf-8")

    print("      Populating SQLite database (data/tuneai.db)...")
    init_db()
    session = SessionLocal()
    try:
        # Clear existing songs table
        session.query(Song).delete()

        # Batch insert songs
        records = []
        for _, row in df.iterrows():
            song = Song(
                track_id=str(row["track_id"]),
                track_name=str(row["track_name"]),
                artist=str(row["artist"]),
                album_name=str(row["album_name"]),
                release_date=str(row.get("release_date", row["release_year"])),
                release_year=int(row["release_year"]),
                genre=str(row["genre"]),
                popularity=float(row["popularity"]),
                duration_ms=float(row["duration_ms"]),
                danceability=float(row["danceability"]),
                energy=float(row["energy"]),
                loudness=float(row["loudness"]),
                speechiness=float(row["speechiness"]),
                acousticness=float(row["acousticness"]),
                instrumentalness=float(row["instrumentalness"]),
                liveness=float(row["liveness"]),
                valence=float(row["valence"]),
                tempo=float(row["tempo"]),
                explicit=bool(row["explicit"]),
                preview_url=row["preview_url"] if pd.notna(row["preview_url"]) else None
            )
            records.append(song)

        session.bulk_save_objects(records)
        session.commit()
        print(f"      Successfully inserted {len(records):,} songs into SQLite.")
    except Exception as e:
        session.rollback()
        print(f"      Error populating database: {e}")
        raise
    finally:
        session.close()

    print("=" * 60)
    print("             PREPROCESSING COMPLETED SUCCESSFULLY")
    print(f"             Final songs processed: {len(df):,}")
    print("=" * 60)

    return df


if __name__ == "__main__":
    preprocess_data()
