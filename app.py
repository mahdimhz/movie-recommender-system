import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# --------------------------------------------------
# 🎬 Page Setup
# --------------------------------------------------
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Movie Recommender System (Content-Based)")
st.write("Pick a movie, and we'll show you similar ones based on genres.")

# --------------------------------------------------
# 📊 Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("data/movies.csv")
    return movies

movies_df = load_data()

# --------------------------------------------------
# 🧠 Build Recommender Model
# --------------------------------------------------
@st.cache_resource
def build_model(data):
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'), stop_words='english')
    genre_matrix = vectorizer.fit_transform(data['genres'])
    similarity = cosine_similarity(genre_matrix)
    return similarity

similarity_matrix = build_model(movies_df)

# --------------------------------------------------
# 🎯 User Input & Recommendations
# --------------------------------------------------
movie_titles = movies_df['title'].tolist()
selected_movie = st.selectbox("🎞 Select a Movie", movie_titles)

if st.button("🔍 Show Recommendations"):
    idx = movies_df[movies_df['title'] == selected_movie].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # top 5, excluding the movie itself

    st.subheader("📽️ You might also like:")
    for i, (movie_idx, score) in enumerate(sim_scores):
        movie_title = movies_df.iloc[movie_idx]['title']
        st.write(f"{i+1}. {movie_title}  ⭐ Similarity: {score:.2f}")
