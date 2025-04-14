# 🎬 Movie Recommender System  
by [Mahdi](https://github.com/mahdimhz) • [LinkedIn](https://www.linkedin.com/in/mahdimhz)

A smart, interactive **content-based movie recommender** built with **Streamlit**, **pandas**, and **scikit-learn**.  
Just type a movie name — and get intelligent recommendations with posters and similarity scores.

👉 **Live Demo:** [Try it now](https://mahdimhz-movie-recommender-system.streamlit.app)

---

## 🔍 How It Works

- Start typing a movie (e.g., `dark`, `lord`, `god`)
- The app suggests close matches using **fuzzy + substring search**
- Select your movie — and see a **poster + top recommendations**
- Click **"Show more recommendations"** to view more similar titles

### 🎞 Poster Support
Posters are fetched live using **TMDb API**, enhancing user experience.

### 🔢 Similarity Score
Each result includes a **similarity score** from `0.0` to `1.0`:
- `1.00` → Perfect genre match
- `0.00` → No genre overlap  
Based on **cosine similarity** of genre vectors.

---

## 🧠 Technologies Used

- Python 3
- [Streamlit](https://streamlit.io)
- Pandas
- scikit-learn (`CountVectorizer`, `cosine_similarity`)
- difflib (`get_close_matches`)
- TMDb API for movie posters

---

## 🛠️ Run Locally

```bash
git clone https://github.com/mahdimhz/movie-recommender-system.git
cd movie-recommender-system
pip install -r requirements.txt
streamlit run app.py
