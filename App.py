import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindCheck · Student Mental Health",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background + text ── */
.stApp {
    background: #0f1117;
    color: #e2e8f0;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Custom hero banner ── */
.hero {
    background: linear-gradient(135deg, #1a1f2e 0%, #151b2a 60%, #0f1117 100%);
    border: 1px solid #2d3748;
    border-radius: 16px;
    padding: 2.5rem 2.5rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0 0 0.4rem;
    letter-spacing: -0.5px;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1rem;
    margin: 0;
}
.pill {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    color: #818cf8;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 3px 12px;
    border-radius: 100px;
    border: 1px solid rgba(99,102,241,0.3);
    margin-bottom: 0.9rem;
    letter-spacing: 0.4px;
    text-transform: uppercase;
}

/* ── Metric cards ── */
.metric-card {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    text-align: center;
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #818cf8;
    line-height: 1;
}
.metric-label {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Section headers ── */
.section-head {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #cbd5e1;
    margin: 1.5rem 0 0.7rem;
    border-left: 3px solid #6366f1;
    padding-left: 0.65rem;
}

/* ── Score gauge label ── */
.score-ring-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #1a1f2e;
    padding: 6px;
    border-radius: 10px;
    border: 1px solid #2d3748;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 600;
    color: #64748b;
    padding: 0.45rem 1.1rem;
}
.stTabs [aria-selected="true"] {
    background: #6366f1 !important;
    color: #fff !important;
}

/* ── Sliders + selects ── */
.stSlider [data-testid="stThumbValue"] { color: #818cf8; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.6rem 1.5rem;
    transition: opacity 0.15s;
}
.stButton > button:hover { opacity: 0.88; }

/* ── Plot containers ── */
.plot-card {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 1.1rem 1.2rem 0.7rem;
    margin-bottom: 0.5rem;
}

/* ── Input labels ── */
label { color: #94a3b8 !important; font-size: 0.88rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib global theme ───────────────────────────────────────────────────
DARK_BG   = "#1a1f2e"
GRID_COL  = "#2d3748"
TEXT_COL  = "#94a3b8"
ACCENT    = "#6366f1"
ACCENT2   = "#8b5cf6"
PALETTE   = ["#6366f1", "#8b5cf6", "#a78bfa", "#38bdf8", "#34d399", "#fb923c"]

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    DARK_BG,
    "axes.edgecolor":    GRID_COL,
    "axes.labelcolor":   TEXT_COL,
    "axes.titlecolor":   TEXT_COL,
    "xtick.color":       TEXT_COL,
    "ytick.color":       TEXT_COL,
    "grid.color":        GRID_COL,
    "text.color":        TEXT_COL,
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("models/model_v1.joblib")
    feature_columns = joblib.load("models/feature_columns_v1.joblib")
    return model, feature_columns

try:
    model, feature_columns = load_model()
    model_loaded = True
except Exception:
    model_loaded = False

# ── Load dataset ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/Student Social Media And Mental Health Impact.csv")

try:
    df = load_data()
    data_loaded = True
except FileNotFoundError:
    df = None
    data_loaded = False

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="pill">Research Tool</div>
  <div class="hero-title">🧠 MindCheck</div>
  <p class="hero-sub">Explore how social media, sleep, and study habits connect to student mental health.</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_dashboard = st.tabs(["🔮  Predict Score", "📊  Dashboard"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 · PREDICT
# ════════════════════════════════════════════════════════════════════════════════
with tab_predict:
    if not model_loaded:
        st.error("Model files (`model_v1.joblib` / `feature_columns_v1.joblib`) not found. "
                 "Place them in the same directory as this script.")
        st.stop()

    st.markdown('<p class="section-head">Your Details</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        age            = st.slider("Age", 15, 30, 20)
        gender         = st.selectbox("Gender", ["Male", "Female", "Other"])
        country        = st.text_input("Country", "USA")
        academic_level = st.selectbox("Academic Level",
                                      ["High School", "Undergraduate", "Graduate"])
        platform       = st.selectbox("Most Used Platform",
                                      ["Instagram", "TikTok", "Facebook",
                                       "YouTube", "Twitter", "Line", "Other"])
        purpose        = st.selectbox("Purpose of Use",
                                      ["Entertainment", "Social Interaction",
                                       "Education", "News", "Other"])

    with col2:
        usage_hours    = st.slider("Social Media (hrs/day)",  0.0, 14.0, 4.0, 0.5)
        study_hours    = st.slider("Study Hours per Day",      0.0, 12.0, 3.0, 0.5)
        physical_hours = st.slider("Physical Activity (hrs/day)", 0.0, 10.0, 2.0, 0.5)
        sleep_hours    = st.slider("Sleep Hours per Night",    2.0, 12.0, 7.0, 0.5)
        stress_level   = st.selectbox("Stress Level", ["Low", "Medium", "High"])

    st.markdown("")
    if st.button("Predict Mental Health Score", type="primary"):
        input_dict = {
            "Age": age, "Gender": gender, "Country": country,
            "Academic_Level": academic_level, "Most_Used_Platform": platform,
            "Purpose_Of_Use": purpose, "Avg_Daily_Usage_Hours": usage_hours,
            "Study_Hours": study_hours, "Physical_Activity_Hours": physical_hours,
            "Sleep_Hours_Per_Night": sleep_hours, "Stress_Level": stress_level,
        }
        input_df   = pd.DataFrame([input_dict])[feature_columns]
        prediction = model.predict(input_df)[0]
        score      = float(np.clip(prediction, 0, 10))

        # ── Gauge chart ──────────────────────────────────────────────────────
        st.markdown('<p class="section-head">Your Result</p>', unsafe_allow_html=True)
        g_col, t_col = st.columns([1, 1], gap="large")

        with g_col:
            fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw=dict(aspect="equal"))
            fig.patch.set_facecolor(DARK_BG)
            theta    = np.linspace(np.pi, 0, 300)
            # Track
            ax.plot(np.cos(theta), np.sin(theta), lw=16, color="#2d3748",
                    solid_capstyle="round")
            # Fill
            fill_end = np.pi - (score / 10) * np.pi
            theta_f  = np.linspace(np.pi, fill_end, 300)
            color    = "#34d399" if score >= 7 else "#fb923c" if score >= 4 else "#f87171"
            ax.plot(np.cos(theta_f), np.sin(theta_f), lw=16, color=color,
                    solid_capstyle="round")
            ax.text(0, -0.15, f"{score:.1f}", ha="center", va="center",
                    fontsize=36, fontweight="bold", color="#f1f5f9",
                    fontfamily="sans-serif")
            ax.text(0, -0.55, "out of 10", ha="center", va="center",
                    fontsize=10, color=TEXT_COL)
            ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.7, 1.15)
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)

        with t_col:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if score >= 7:
                st.success(f"**Score: {score:.1f} / 10 — Healthy Range 🙂**\n\n"
                           "Your habits seem well-balanced. Keep maintaining your sleep, "
                           "activity, and study routines.")
            elif score >= 4:
                st.warning(f"**Score: {score:.1f} / 10 — Moderate Range ⚠️**\n\n"
                           "Some habits may be worth adjusting. Try reducing screen time, "
                           "improving sleep, or adding short activity breaks.")
            else:
                st.error(f"**Score: {score:.1f} / 10 — Needs Attention 🔴**\n\n"
                         "Consider speaking with a counselor or trusted person. Small changes "
                         "in sleep and physical activity can have a meaningful impact.")

        # ── Key factor breakdown ──────────────────────────────────────────────
        st.markdown('<p class="section-head">How Your Inputs Compare to the Dataset Average</p>',
                    unsafe_allow_html=True)
        if data_loaded:
            factors = {
                "Sleep Hours":        (sleep_hours, df["Sleep_Hours_Per_Night"].mean()),
                "Study Hours":        (study_hours, df["Study_Hours"].mean()),
                "Physical Activity":  (physical_hours, df["Physical_Activity_Hours"].mean()),
                "Social Media (hrs)": (usage_hours, df["Avg_Daily_Usage_Hours"].mean()),
            }
            fig, ax = plt.subplots(figsize=(7, 2.6))
            labels  = list(factors.keys())
            yours   = [v[0] for v in factors.values()]
            avg     = [v[1] for v in factors.values()]
            x = np.arange(len(labels))
            bars_a = ax.bar(x - 0.2, avg,   0.38, label="Dataset Avg",  color=GRID_COL, zorder=2)
            bars_y = ax.bar(x + 0.2, yours, 0.38, label="You",          color=ACCENT,   zorder=2, alpha=0.9)
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
            ax.legend(fontsize=8, framealpha=0)
            ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
            st.pyplot(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 · DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    if not data_loaded:
        st.warning("Dataset CSV not found. Add it and restart the app.")
        st.stop()

    # ── Summary metrics ───────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    metrics = [
        ("Students",           f"{len(df):,}"),
        ("Avg Mental Health",  f"{df['Mental_Health_Score'].mean():.1f}"),
        ("Avg Social Media",   f"{df['Avg_Daily_Usage_Hours'].mean():.1f} h"),
        ("Avg Sleep",          f"{df['Sleep_Hours_Per_Night'].mean():.1f} h"),
        ("Avg Study",          f"{df['Study_Hours'].mean():.1f} h"),
    ]
    for col, (label, value) in zip([m1, m2, m3, m4, m5], metrics):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{value}</div>
          <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ════ ROW 1: Distribution + Heatmap ══════════════════════════════════════
    st.markdown('<p class="section-head">Score Distribution & Feature Correlations</p>',
                unsafe_allow_html=True)
    r1c1, r1c2 = st.columns(2, gap="medium")

    with r1c1:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        sns.histplot(df["Mental_Health_Score"], bins=30, kde=True, ax=ax,
                     color=ACCENT, alpha=0.7, line_kws={"lw": 2, "color": "#a78bfa"})
        ax.set_xlabel("Mental Health Score"); ax.set_ylabel("Count")
        ax.set_title("Score Distribution", fontsize=11, pad=10)
        ax.yaxis.grid(True, alpha=0.25); ax.set_axisbelow(True)
        st.pyplot(fig, use_container_width=True)

    with r1c2:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        num_df = df.select_dtypes(include="number")
        mask   = np.triu(np.ones_like(num_df.corr(), dtype=bool))
        sns.heatmap(num_df.corr(), mask=mask, cmap="coolwarm", annot=True,
                    fmt=".2f", ax=ax, annot_kws={"size": 7},
                    linewidths=0.5, linecolor="#0f1117",
                    cbar_kws={"shrink": 0.8})
        ax.set_title("Correlation Matrix", fontsize=11, pad=10)
        st.pyplot(fig, use_container_width=True)

    # ════ ROW 2: Scatter plots ════════════════════════════════════════════════
    st.markdown('<p class="section-head">Key Lifestyle Factors vs Mental Health</p>',
                unsafe_allow_html=True)
    r2c1, r2c2, r2c3 = st.columns(3, gap="medium")

    scatter_cfg = [
        ("Sleep_Hours_Per_Night",    "Sleep Hours / Night"),
        ("Avg_Daily_Usage_Hours",    "Social Media Hours / Day"),
        ("Physical_Activity_Hours",  "Physical Activity Hours / Day"),
    ]
    scatter_colors = [ACCENT, "#38bdf8", "#34d399"]

    for col, (xcol, xlabel), color in zip([r2c1, r2c2, r2c3], scatter_cfg, scatter_colors):
        with col:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.scatter(df[xcol], df["Mental_Health_Score"],
                       alpha=0.3, s=14, color=color, linewidths=0)
            # trend line
            m, b = np.polyfit(df[xcol].dropna(), df["Mental_Health_Score"][df[xcol].notna()], 1)
            xr = np.linspace(df[xcol].min(), df[xcol].max(), 100)
            ax.plot(xr, m*xr + b, color="white", lw=1.5, alpha=0.7)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel("Mental Health Score", fontsize=9)
            ax.yaxis.grid(True, alpha=0.25); ax.set_axisbelow(True)
            st.pyplot(fig, use_container_width=True)

    # ════ ROW 3: Box plots by categorical ════════════════════════════════════
    st.markdown('<p class="section-head">Mental Health by Stress, Gender & Academic Level</p>',
                unsafe_allow_html=True)
    r3c1, r3c2, r3c3 = st.columns(3, gap="medium")

    box_cfgs = [
        ("Stress_Level",    ["Low", "Medium", "High"],        "Stress Level"),
        ("Gender",          None,                              "Gender"),
        ("Academic_Level",  ["High School","Undergraduate","Graduate"], "Academic Level"),
    ]

    for col, (grp, order, xlabel) in zip([r3c1, r3c2, r3c3], box_cfgs):
        with col:
            fig, ax = plt.subplots(figsize=(4, 3))
            cats = [o for o in (order or df[grp].unique()) if o in df[grp].unique()]
            data_list = [df[df[grp] == c]["Mental_Health_Score"].dropna() for c in cats]
            bp = ax.boxplot(data_list, patch_artist=True, tick_labels=cats,
                            medianprops=dict(color="white", linewidth=2),
                            whiskerprops=dict(color=TEXT_COL),
                            capprops=dict(color=TEXT_COL),
                            flierprops=dict(marker="o", markersize=3,
                                            markerfacecolor=TEXT_COL, alpha=0.3))
            for patch, color in zip(bp["boxes"], PALETTE):
                patch.set_facecolor(color); patch.set_alpha(0.6)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel("Mental Health Score", fontsize=9)
            ax.yaxis.grid(True, alpha=0.25); ax.set_axisbelow(True)
            ax.tick_params(axis="x", labelsize=8)
            st.pyplot(fig, use_container_width=True)

    # ════ ROW 4: Platform bar + Purpose pie ══════════════════════════════════
    st.markdown('<p class="section-head">Platform & Usage Purpose Breakdown</p>',
                unsafe_allow_html=True)
    r4c1, r4c2 = st.columns([1.5, 1], gap="medium")

    with r4c1:
        plat_avg = (df.groupby("Most_Used_Platform")["Mental_Health_Score"]
                      .mean().sort_values(ascending=True))
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        bars = ax.barh(plat_avg.index, plat_avg.values, color=ACCENT, alpha=0.8, height=0.6)
        for bar, val in zip(bars, plat_avg.values):
            ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}", va="center", fontsize=8, color=TEXT_COL)
        ax.set_xlabel("Avg Mental Health Score"); ax.set_xlim(0, plat_avg.max() + 8)
        ax.set_title("Avg Score by Platform", fontsize=11, pad=10)
        ax.xaxis.grid(True, alpha=0.25); ax.set_axisbelow(True)
        st.pyplot(fig, use_container_width=True)

    with r4c2:
        purpose_counts = df["Purpose_Of_Use"].value_counts()
        fig, ax = plt.subplots(figsize=(4, 3.2))
        wedges, texts, autotexts = ax.pie(
            purpose_counts,
            labels=purpose_counts.index,
            autopct="%1.0f%%",
            colors=PALETTE[:len(purpose_counts)],
            startangle=90,
            pctdistance=0.78,
            wedgeprops=dict(linewidth=1.5, edgecolor=DARK_BG),
        )
        for t in texts: t.set_fontsize(8); t.set_color(TEXT_COL)
        for at in autotexts: at.set_fontsize(7.5); at.set_color("white")
        ax.set_title("Purpose of Use", fontsize=11, pad=10)
        st.pyplot(fig, use_container_width=True)

    # ════ ROW 5: Usage hours distribution + Score by study hours heatmap ═════
    st.markdown('<p class="section-head">Usage Patterns & Study Heatmap</p>',
                unsafe_allow_html=True)
    r5c1, r5c2 = st.columns(2, gap="medium")

    with r5c1:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        for stress, color in zip(["Low","Medium","High"],["#34d399","#fb923c","#f87171"]):
            subset = df[df["Stress_Level"] == stress]["Avg_Daily_Usage_Hours"]
            sns.kdeplot(subset, ax=ax, color=color, fill=True, alpha=0.25, label=stress, linewidth=1.5)
        ax.set_xlabel("Social Media Hours / Day")
        ax.set_ylabel("Density")
        ax.set_title("Usage Distribution by Stress Level", fontsize=11, pad=10)
        ax.legend(title="Stress", fontsize=8, title_fontsize=8, framealpha=0)
        ax.yaxis.grid(True, alpha=0.25); ax.set_axisbelow(True)
        st.pyplot(fig, use_container_width=True)

    with r5c2:
        # Bin sleep and study, show avg mental health as heatmap
        df["sleep_bin"] = pd.cut(df["Sleep_Hours_Per_Night"],
                                  bins=[0,5,6,7,8,13],
                                  labels=["<5","5-6","6-7","7-8","8+"])
        df["study_bin"] = pd.cut(df["Study_Hours"],
                                  bins=[0,2,4,6,8,13],
                                  labels=["<2","2-4","4-6","6-8","8+"])
        pivot = df.pivot_table(values="Mental_Health_Score",
                                index="sleep_bin", columns="study_bin", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        sns.heatmap(pivot, cmap="RdYlGn", annot=True, fmt=".0f", ax=ax,
                    linewidths=0.5, linecolor="#0f1117",
                    cbar_kws={"shrink": 0.85},
                    annot_kws={"size": 8})
        ax.set_xlabel("Study Hours / Day", fontsize=9)
        ax.set_ylabel("Sleep Hours / Night", fontsize=9)
        ax.set_title("Avg Score: Sleep × Study", fontsize=11, pad=10)
        st.pyplot(fig, use_container_width=True)

    # ════ ROW 6: Top countries (if available) ════════════════════════════════
    if "Country" in df.columns and df["Country"].nunique() > 1:
        st.markdown('<p class="section-head">Mental Health Score by Country (Top 10)</p>',
                    unsafe_allow_html=True)
        country_avg = (df.groupby("Country")["Mental_Health_Score"]
                         .agg(["mean","count"])
                         .query("count >= 5")
                         .sort_values("mean", ascending=False)
                         .head(10))
        fig, ax = plt.subplots(figsize=(10, 3))
        colors  = [ACCENT if i < 3 else GRID_COL for i in range(len(country_avg))]
        ax.bar(country_avg.index, country_avg["mean"], color=colors, alpha=0.85, zorder=2)
        ax.yaxis.grid(True, alpha=0.25); ax.set_axisbelow(True)
        ax.set_ylabel("Avg Mental Health Score"); ax.set_xlabel("Country")
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        # annotate
        for i, (idx, row) in enumerate(country_avg.iterrows()):
            ax.text(i, row["mean"] + 0.3, f"{row['mean']:.1f}",
                    ha="center", fontsize=7.5, color=TEXT_COL)
        top3_patch = mpatches.Patch(color=ACCENT, label="Top 3")
        rest_patch  = mpatches.Patch(color=GRID_COL, label="Others")
        ax.legend(handles=[top3_patch, rest_patch], fontsize=8, framealpha=0)
        st.pyplot(fig, use_container_width=True)
