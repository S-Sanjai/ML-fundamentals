# Movie Recommendation System

A content-based movie recommendation system built from scratch using custom K-Nearest Neighbors (KNN) implementation with TF-IDF vectorization. This project demonstrates core machine learning concepts without relying on high-level ML libraries.

## 🎯 Overview

This is the machine learning backend of a movie recommendation system. It analyzes movie overviews and genres to find similar movies using a custom-built KNN algorithm and TF-IDF text vectorization implemented from scratch.

## ✨ Key Features

- **Custom KNN Algorithm**: Built from scratch without using sklearn's KNN
- **TF-IDF from First Principles**: Understanding and implementing text vectorization
- **Genre Multi-Hot Encoding**: Binary representation of movie genres
- **Cosine Similarity**: Distance metric for finding similar movies
- **Pre-computed Feature Matrix**: Optimized for fast recommendations

## 📁 Project Structure

```
04-Movie-Recommendation-System/
├── data/
│   └── movies.csv                    # Movie dataset with titles, overviews, genres
├── models/
│   ├── final_matrix.npz              # Pre-computed feature matrix for all movies
│   ├── tfidf_vectorizer.joblib       # Trained TF-IDF vectorizer
│   └── feature_names.npy             # Feature names for interpretability
├── notebooks/
│   └── data_prep.ipynb               # Data preprocessing and model training
└── src/
    ├── __init__.py
    ├── knn_scratch.py                # Custom KNN implementation
    ├── tf_idf.py                     # TF-IDF vectorizer from scratch
    ├── api_auth.py                   # TMDB API credentials (optional)
    └── scrape_data.py                # Data collection utilities (optional)
```

## 🔬 How It Works

### 1. Data Preprocessing

The `notebooks/data_prep.ipynb` notebook performs the following steps:

- **Load Dataset**: Reads `movies.csv` containing movie metadata
- **Text Vectorization**: Converts movie overviews into TF-IDF vectors
  - Calculates term frequency (TF) for each word in a document
  - Calculates inverse document frequency (IDF) across all documents
  - Combines TF and IDF to create meaningful text representations
- **Genre Encoding**: Creates multi-hot encoded vectors for genres
  - Each genre gets a binary feature (1 if present, 0 if absent)
- **Feature Combination**: Concatenates TF-IDF and genre vectors
  - Combined vector represents the movie's content profile

### 2. K-Nearest Neighbors Algorithm

The `src/knn_scratch.py` file contains the core recommendation logic:

```python
from src.knn_scratch import MovieRecommender

# Initialize recommender
recommender = MovieRecommender('data/movies.csv', 'models/final_matrix.npz')

# Get recommendations
recommendations = recommender.get_recommendations('Inception', k=5)
```

**Algorithm Steps**:
1. Find the feature vector for the input movie
2. Calculate cosine similarity with all other movies
3. Sort by similarity score (highest first)
4. Return top-k most similar movies

**Cosine Similarity Formula**:
```
similarity = (A · B) / (||A|| × ||B||)
```

### 3. TF-IDF Vectorization

The `src/tf_idf.py` implements TF-IDF from scratch:

- **Type**: Content-Based Filtering
- **Technique**: Term Frequency-Inverse Document Frequency
- **Goal**: Convert text descriptions into numerical vectors

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd 04-Movie-Recommendation-System
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy scipy joblib jupyter requests
   ```

3. **Prepare the data**
   ```bash
   jupyter notebook notebooks/data_prep.ipynb
   ```
   Run all cells to generate the model files.

### Usage

```python
from src.knn_scratch import MovieRecommender

# Initialize
recommender = MovieRecommender(
    data_path='data/movies.csv',
    matrix_path='models/final_matrix.npz'
)

# Get recommendations
recs = recommender.get_recommendations('The Dark Knight', k=5)
print(recs)
```

## 📝 License

This project is for educational purposes as part of ML fundamentals learning.
