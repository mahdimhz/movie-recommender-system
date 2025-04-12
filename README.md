# 🎬 Movie Recommender System

This project builds a Netflix-style movie recommendation system using real-world data and collaborative filtering.

It uses the MovieLens dataset to train a model that can predict how users might rate movies they haven't seen — and recommend new films.

---

## 📌 What It Does

- Loads and explores the MovieLens 100k dataset
- Analyzes most-rated and top-rated movies
- Builds a collaborative filtering model using SVD from the Surprise library
- Recommends top movies to each user based on their past behavior

---

## 💻 Tools Used

- Python
- Jupyter Notebook
- Pandas, Matplotlib
- Surprise (SVD recommender)

---

## 🧠 Highlights

- RMSE: ~0.87  
- MAE: ~0.67  
- 100k+ ratings, 9k+ movies, 610 users

---

## 🔍 Example: Recommendations for User 1

- Shawshank Redemption
- The Departed
- The Dark Knight Rises
- Wallace & Gromit
- The Thing

---

## 📈 Bonus Analysis: Forrest Gump?

Forrest Gump was the most rated movie in the dataset — but didn’t appear in the top 10.  
It had a high average rating (4.16) but not high *enough* to beat others that scored 4.23+.

---

## 🚀 What's Next

- Try other algorithms (like KNN or SVD++)
- Add genre-based recommendations
- Build a simple web app using Streamlit

---

## 🙋‍♂️ About Me

I'm Mahdi, a data science master's student passionate about machine learning and building practical AI tools.

👉 [GitHub](https://github.com/mahdimhz)  
👉 [LinkedIn](https://www.linkedin.com/in/mahdimhz)
