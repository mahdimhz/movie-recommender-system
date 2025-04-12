import streamlit as st
import pandas as pd
from surprise import SVD, Dataset, Reader

# --------------------------------------------------
# 🎬 Streamlit Page Setup
# --------------------------------------------------
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Movie Recommender System")
st.write("Enter a user ID below to get personalized movie recommendations.")

# --------------------------------------------------
# 📥 User Input
# --------------------------------------------------
user_id = st.number_input("Enter User ID (1–610)", min_value=1, max_value=610, step=1)
show_recommend = st.button("🎯 Get Recommendations")

# --------------------------------------------------
# 📊 Load & Prepare Data
# --------------------------------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    df = pd.merge(ratings, movies, on="movieId")
    return df

df = load_data()

# --------------------------------------------------
# 🧠 Train Model (SVD)
# --------------------------------------------------
@st.cache_resource
def train_model(dataframe):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(dataframe[["userId", "movieId", "rating"]], reader)
    trainset = data.build_full_trainset()
    model = SVD()
    model.fit(trainset)
    return model

model = train_model(df)

# --------------------------------------------------
# 🎯 Generate Recommendations
# --------------------------------------------------
if show_recommend:
    # Movies the user has already rated
    rated_movie_ids = df[df['userId'] == user_id]['movieId'].tolist()
    all_movie_ids = df['movieId'].unique()
    movies_to_predict = [mid for mid in all_movie_ids if mid not in rated_movie_ids]

    # Predict ratings
    predictions = [model.predict(user_id, movie_id) for movie_id in movies_to_predict]
    top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:5]

    # Movie titles and predicted ratings
    top_movie_ids = [pred.iid for pred in top_predictions]
    top_movies = df[df['movieId'].isin(top_movie_ids)][['movieId', 'title']].drop_duplicates()
    top_movies['Predicted Rating'] = [round(pred.est, 2) for pred in top_predictions]

    # Show results
    st.subheader("📽️ Top Recommended Movies:")
    st.table(top_movies[['title', 'Predicted Rating']].reset_index(drop=True))
