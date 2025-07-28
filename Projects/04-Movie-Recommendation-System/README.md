# Movie Recommendation System

This project is a content-based movie recommendation system that uses K-Nearest Neighbors (KNN) to find similar movies. It recommends movies based on their overview and genres.

## Project Structure

The project is organized into the following directories:

- `data/`: Contains the raw movie dataset (`movies.csv`).
- `notebooks/`: Contains the Jupyter notebook for data preprocessing (`data_prep.ipynb`).
- `models/`: Contains the saved model files, including the TF-IDF vectorizer, the final feature matrix, and feature names.
- `src/`: Contains the Python source code for the project.

## How It Works

The recommendation system works by converting movie information into numerical vectors and then finding the most similar movies using the K-Nearest Neighbors algorithm.

1.  **Data Preprocessing**:
    -   **Movie Overviews**: The text from the 'overview' of each movie is converted into a numerical vector using TF-IDF (Term Frequency-Inverse Document Frequency).
    -   **Movie Genres**: The movie genres are multi-hot encoded. This means that for each movie, a binary vector is created where each element corresponds to a genre, and its value is 1 if the movie has that genre and 0 otherwise.

2.  **Feature Combination**: The TF-IDF vector for the overview and the multi-hot encoded genre vector are combined to create a single feature vector for each movie.

3.  **Recommendation**: The K-Nearest Neighbors (KNN) algorithm is used to find the 'k' most similar movies to a given movie based on the cosine similarity of their feature vectors.

## How to Use

1.  **Data Preparation**: Run the `data_prep.ipynb` notebook in the `notebooks` directory to preprocess the data and create the necessary model files.
2.  **Get Recommendations**: Use the scripts in the `src` directory to get movie recommendations. The `knn_scratch.py` file contains the core KNN recommendation logic.
