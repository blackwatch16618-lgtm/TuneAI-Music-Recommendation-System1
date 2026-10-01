# TuneAI 🎵

TuneAI is a hybrid music recommendation engine that combines Matrix Factorization using SVD with content-based cosine similarity to recommend songs based on listening behavior and musical features.

> **Suggested Tagline:** *Discover your next favorite song with TuneAI.*

---

## Features

- 🎵 **Song recommendation**: Instant recommendations when selecting any track in the catalog.
- 👤 **User-based personalization**: Adapts recommendations to listener history using collaborative filtering.
- 🧠 **SVD Matrix Factorization**: Uncovers latent dimensions of user preferences and track characteristics.
- 🎧 **Content-based recommendation**: Calculates acoustic proximity over multi-dimensional audio feature vectors.
- 🔎 **Cosine similarity**: Compares musical attributes (tempo, energy, valence, danceability, acousticness).
- 📊 **Recommendation analytics**: High-resolution dataset charts and statistical breakdowns.
- 📅 **2022–2024 song filtering**: Strictly restricted to modern releases (2022, 2023, 2024).
- 🎼 **Genre filtering**: Filter recommendations dynamically across real dataset genres.
- 🔥 **Popular songs**: Browse top-streamed and highest-rated tracks.
- ⚡ **FastAPI backend**: High-performance RESTful API with automated docs (`/docs`).
- 💻 **Responsive web interface**: Clean, dark music streaming interface (Spotify/Tidal inspired).

---

## Architecture

```
       Dataset (Real Spotify 2022–2024 Tracks)
                          ↓
                    Preprocessing (preprocess.py)
                          ↓
               User-Song Interaction Matrix (CSR)
                          ↓
              SVD Matrix Factorization (svd_model.py)
                          ↓
              Latent Factors (Users & Songs)
                          ↓
                    Candidate Generation
                          ↓
             Cosine Similarity (similarity.py)
                          ↓
               Hybrid Ranking (recommender.py)
                          ↓
               TuneAI Web Application (app.py)
```

---

## Machine Learning

TuneAI fuses collaborative filtering with content-based filtering in a weighted hybrid architecture:

### 1. Matrix Factorization (SVD)
SVD learns latent relationships between users and songs.
Given a sparse user-song interaction matrix $R \in \mathbb{R}^{U \times I}$, TruncatedSVD factorizes the interaction matrix into $k = 50$ latent factor matrices:
$$R \approx U \cdot \Sigma \cdot V^T$$
- **User Factors ($P$)**: Dense representation of listener tastes.
- **Song Factors ($Q$)**: Dense representation of songs in preference space.
- **Predicted Preference Score**: $\hat{r}_{u,i} = P_u \cdot Q_i^T$. Scores are normalized to $[0, 1]$.

### 2. Cosine Similarity
Cosine similarity compares the musical feature vectors of songs.
Each song is characterized by 10 continuous audio attributes:
$$\mathbf{x} = [\text{danceability}, \text{energy}, \text{loudness}, \text{speechiness}, \text{acousticness}, \text{instrumentalness}, \text{liveness}, \text{valence}, \text{tempo}, \text{duration\_ms}]$$
Features are standardized using `StandardScaler` to have zero mean and unit variance. For target song $\mathbf{u}$ and candidate song $\mathbf{v}$:
$$\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$
Cosine scores are rescaled from $[-1, 1]$ to $[0, 1]$ and displayed as a percentage match (e.g. `91% Match`). The selected song itself is strictly excluded.

### 3. Hybrid Model
$$\text{Final Score} = 0.70 \times \text{SVD Score} + 0.30 \times \text{Content Similarity}$$

**Why the two methods are combined:**
- **Collaborative filtering (SVD)** excels at capturing communal listening patterns and discovering unexpected musical connections that feature vectors alone cannot detect.
- **Content-based similarity** ensures acoustic harmony and solves the cold-start challenge: if a listener is new or a song has few interaction logs, the system seamlessly falls back to audio similarity without degradation.

---

## Dataset

- **Dataset Name**: Spotify Track Audio Features & Albums Dataset with 2024 Billboard Hot 100 Augmentation
- **Original Source**: Extracted via the official Spotify Web API, archived on Kaggle (`tonygordonjr/spotify-dataset-2023`) and GitHub (`jroblar/cse6242-project`).
- **License**: Community Data License Agreement – Permissive (CDLA-Permissive-1.0) / MIT.
- **Download Date**: September 2026 / Archived 2023–2024.
- **Original Rows**: 375,241 tracks.
- **Processed Rows**: **89,853 unique tracks**.
- **2022–2024 Restriction**: Every track is strictly filtered such that:
  $$\text{release\_year} \in \{2022, 2023, 2024\}$$
- **Columns Used**: `track_id`, `track_name`, `artist`, `album_name`, `release_date`, `release_year`, `genre`, `popularity`, `duration_ms`, `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `explicit`, `user_id`, `listen_count`.

### Implicit Rating Mapping
Listening counts are mapped into implicit ratings via configurable frequency tiers:
- `listen_count = 1` $\rightarrow$ `rating = 1`
- `listen_count = 2–4` $\rightarrow$ `rating = 2`
- `listen_count = 5–9` $\rightarrow$ `rating = 3`
- `listen_count = 10–19` $\rightarrow$ `rating = 4`
- `listen_count >= 20` $\rightarrow$ `rating = 5`

*Note: These are derived implicit scores based on listening frequency, not explicit user survey ratings.*

---

## Project Structure

```
TuneAI/
│
├── app.py                      # FastAPI web server and routing
├── config.py                   # Central configurations, paths, hyperparameters
├── preprocess.py               # Data cleaning, year filtering (2022-2024), SQLite population
├── train_model.py              # SVD training, scaler fitting, artifact serialization
├── recommender.py              # Hybrid recommendation engine orchestrator
├── svd_model.py                # TruncatedSVD collaborative filtering implementation
├── similarity.py               # Audio feature StandardScaler & cosine similarity
├── evaluate_model.py           # Precision@K, Recall@K, Hit Rate@K evaluation
├── analysis.py                 # Matplotlib dataset analytics & chart generation
├── database.py                 # SQLAlchemy SQLite schema and models
├── pytest.ini                  # Pytest configuration
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── data_source.md              # Dataset transparency and licensing details
│
├── data/
│   ├── raw/
│   │   └── songs.csv           # Raw combined music dataset
│   ├── processed/
│   │   └── tuneai_songs_2022_2024.csv # Cleaned 2022–2024 catalog
│   └── tuneai.db               # SQLite database
│
├── models/
│   ├── svd_model.pkl           # Trained SVD latent factors
│   ├── scaler.pkl              # Fitted audio feature StandardScaler
│   ├── song_index.pkl          # Song ID to matrix index mappings
│   └── user_index.pkl          # User ID to matrix index mappings
│
├── static/
│   ├── css/style.css           # Modern music streaming stylesheet
│   ├── js/app.js               # Frontend JavaScript controller
│   ├── images/                 # Image assets
│   └── charts/                 # Generated analytics charts (matplotlib)
│       ├── songs_by_year.png
│       ├── songs_by_genre.png
│       ├── popular_artists.png
│       ├── popular_songs.png
│       └── audio_features_distribution.png
│
├── templates/
│   └── index.html              # Modern music discovery interface
│
└── tests/
    ├── test_preprocessing.py   # Preprocessing & column mapping tests
    ├── test_similarity.py      # Content-based cosine similarity tests
    ├── test_recommender.py     # Hybrid logic & fallback tests
    └── test_api.py             # FastAPI endpoint integration tests
```

---

## Running the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess Data
Loads the raw CSV, cleans metadata, filters to 2022–2024, and populates `data/tuneai.db`:
```bash
python preprocess.py
```

### 3. Train Models
Trains the SVD matrix factorization model and fits the audio feature scaler:
```bash
python train_model.py
```

### 4. Evaluate Model (Optional)
Calculates Precision@K, Recall@K, and Hit Rate@K on holdout test interactions:
```bash
python evaluate_model.py
```

### 5. Generate Data Analysis Charts (Optional)
Produces charts in `static/charts/`:
```bash
python analysis.py
```

### 6. Run the Automated Test Suite
```bash
pytest -v tests/
```

### 7. Start the FastAPI Web Server
```bash
uvicorn app:app --reload
```

Open your browser and navigate to:
```
http://127.0.0.1:8000
```

Interactive OpenAPI documentation is available at:
```
http://127.0.0.1:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web discovery application |
| `GET` | `/songs` | Paginated catalog with `year` and `genre` filters |
| `GET` | `/search?q=query` | Partial-match case-insensitive search |
| `GET` | `/song/{track_id}` | Detailed track metadata and audio attributes |
| `GET` | `/recommendations/{track_id}` | Hybrid recommendations for track and active user |
| `GET` | `/recommendations?user_id=uid` | Personalized recommendations for a user |
| `GET` | `/similar/{track_id}` | Content-based cosine similarity matches |
| `GET` | `/genres` | Unique genres in the catalog |
| `GET` | `/years` | Allowed release years (`[2022, 2023, 2024]`) |
| `GET` | `/popular` | Top popular songs in catalog |
| `GET` | `/users` | Sample active user accounts for interactive testing |
| `GET` | `/analytics` | Dataset summary and statistical breakdown |
