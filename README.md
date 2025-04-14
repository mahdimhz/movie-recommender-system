# 🎬 Movie Recommender System  
by [Mahdi](https://github.com/mahdimhz) • [LinkedIn](https://www.linkedin.com/in/mahdimhz)

This is a content-based movie recommender system built using **Streamlit**, **pandas**, and **scikit-learn**.  
It allows users to search for a movie by name and receive top genre-based recommendations — all in a clean, interactive web app.

👉 **Live Demo:** [Try it now](https://mahdimhz-movie-recommender-system.streamlit.app)

---

## 🔍 How It Works

- Users type part of a movie name (e.g. `dark`, `shaw`, `lord`)
- The app uses **fuzzy + partial matching** to help them find the right title
- Based on selected movie’s genres, it recommends the top 5 most similar movies

### 🔢 Similarity Score
Each recommendation includes a **similarity score** from `0.0` to `1.0`:
- `1.00` → Perfect match (same genre structure)
- `0.00` → No genre overlap  
The score is computed using **cosine similarity** on genre vectors.

---

## 🧠 Technologies Used

- Python 3
- Streamlit
- Pandas
- scikit-learn (`CountVectorizer`, `cosine_similarity`)
- difflib (`get_close_matches`)

---

## 🛠️ How to Run It Locally

```bash
git clone https://github.com/mahdimhz/movie-recommender-system.git
cd movie-recommender-system
pip install -r requirements.txt
streamlit run app.py
