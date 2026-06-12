# 🧠 MindCheck — Student Social Media & Mental Health Impact

A Streamlit-based interactive application that analyzes the relationship between students' social media usage, sleep, study habits, physical activity, and mental health, and predicts a **Mental Health Score** using a trained machine learning model.

---

## ✨ Features

### 🔮 Prediction
- Interactive form to input personal lifestyle details (age, gender, country, academic level, platform usage, study/sleep/activity hours, stress level)
- ML-powered Mental Health Score prediction (0–10) using a pre-trained scikit-learn model
- Gauge chart visualization of predicted score with color-coded ranges (Healthy / Moderate / Needs Attention)
- Comparison of user inputs vs. dataset averages (sleep, study, activity, social media usage)

### 📊 Dashboard / Analytics
- Summary metric cards (total students, average mental health score, average usage, sleep, study hours)
- Mental health score distribution with KDE overlay
- Correlation heatmap of numeric features
- Scatter plots with trend lines (sleep, social media usage, physical activity vs. mental health)
- Box plots by stress level, gender, and academic level
- Platform-wise average mental health score (horizontal bar chart)
- Purpose-of-use breakdown (pie chart)
- Social media usage distribution by stress level (KDE)
- Sleep × Study heatmap of average mental health score
- Top 10 countries by average mental health score

### 🎨 UI/UX
- Custom dark-themed UI with Inter & Space Grotesk fonts
- Tabbed navigation (Predict / Dashboard)
- Custom hero banner, metric cards, and styled charts (Matplotlib + Seaborn themed for dark mode)

---

## 🛠 Tech Stack

| Category | Technology |
|---|---|
| Frontend / UI | Streamlit |
| Backend / App Logic | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | scikit-learn (1.8.0), joblib (model serialization) |
| Visualization | Matplotlib, Seaborn |
| Database | Not detected in repository (data loaded from local CSV file) |
| DevOps Tools | Dev Containers (`.devcontainer/devcontainer.json`) |


---

## 🏗 Architecture Overview

This is a **single-page Streamlit application** (`App.py`) that operates as a self-contained data science/ML demo:

1. **Model Layer** — A pre-trained regression model (`models/model_v1.joblib`) and its feature column schema (`models/feature_columns_v1.joblib`) are loaded via `joblib` and cached using `@st.cache_resource`.
2. **Data Layer** — A static CSV dataset (`data/Student Social Media And Mental Health Impact.csv`) is loaded and cached using `@st.cache_data`.
3. **Presentation Layer** — Streamlit renders two tabs:
   - **Predict Score**: Collects user input via sliders/selectboxes, constructs a feature DataFrame aligned to `feature_columns`, and runs `model.predict()`.
   - **Dashboard**: Generates multiple Matplotlib/Seaborn visualizations directly from the loaded dataset.
4. **Model Training** — A Jupyter Notebook (`Notebook/Student_mental_health_and_socialmedia_updated.ipynb`) presumably contains the exploratory data analysis and model training pipeline used to produce the `.joblib` artifacts (not verified in detail beyond file presence).

No backend API server, database, or external service integrations are present — the application is monolithic and runs entirely within the Streamlit process.

