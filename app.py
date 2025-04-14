import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from difflib import get_close_matches

# --------------------------------------------------
# 🎬 Page Setup
# --------------------------------------------------
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Movie Recommender System (Content-Based)")
st.write("Type a movie name, and we’ll show you similar ones based on genres.")

# --------------------------------------------------
# 📊 Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("data/movies.csv")
    return movies

movies_df = load_data()
movie_titles = movies_df['title'].tolist()

# --------------------------------------------------
# 🧠 Build Model
# --------------------------------------------------
@st.cache_resource
def build_model(data):
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'), stop_words='english')
    genre_matrix = vectorizer.fit_transform(data['genres'])
    similarity = cosine_similarity(genre_matrix)
    return similarity

similarity_matrix = build_model(movies_df)

# --------------------------------------------------
# 🔍 User Search Input
# --------------------------------------------------
search_query = st.text_input("🔎 Type a movie name:")

matched_movie = None
if search_query:
    # Create lowercase mapping
    title_map = {title.lower(): title for title in movie_titles}
    lowered_titles = list(title_map.keys())

    query = search_query.lower()

        # First: Try fuzzy matching
    matches_lower = get_close_matches(query, lowered_titles, n=5, cutoff=0.4)

    # Add substring matches (not limited to 5)
    partial_matches = [title for title in lowered_titles if query in title]

    # Combine both and remove duplicates
    combined_matches = list(dict.fromkeys(matches_lower + partial_matches))


    # Convert back to original titles
    matches = [title_map[match] for match in combined_matches]

    if matches:
        matched_movie = st.selectbox("🎞 Select the closest match:", matches)
    else:
        st.warning("No close match found. Try a different title.")

# --------------------------------------------------
# 🎯 Show Recommendations
# --------------------------------------------------
if matched_movie and st.button("🔍 Show Recommendations"):
    idx = movies_df[movies_df['title'] == matched_movie].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]

    st.subheader("📽️ You might also like:")
    for i, (movie_idx, score) in enumerate(sim_scores):
        st.write(f"{i+1}. {movies_df.iloc[movie_idx]['title']}  ⭐ Similarity: {score:.2f}")
