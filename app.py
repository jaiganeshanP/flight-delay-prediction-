"""
app.py — Flight Delay Prediction System (Streamlit)
Run: streamlit run app.py
"""

import os, warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

from utils import (
    AIRLINES, AIRPORTS, WEATHER_CONDITIONS, DAY_NAMES,
    AIRLINE_NAMES, WEATHER_DELAY_RISK,
    preprocess_input, get_delay_interpretation, simulate_realtime_factors,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark aviation theme */
.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
}

.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid #1e40af;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: "✈";
    position: absolute;
    right: 2rem; top: 50%;
    transform: translateY(-50%);
    font-size: 6rem;
    opacity: 0.06;
}
.main-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    color: #93c5fd;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.5px;
}
.main-header p { color: #94a3b8; margin: 0; font-size: 0.95rem; }

/* Prediction result card */
.result-card {
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}
.result-card h2 { font-family: 'Space Mono', monospace; margin: 0 0 0.5rem 0; }
.result-card p  { margin: 0; color: #cbd5e1; font-size: 0.9rem; }

/* Metric tiles */
.metric-tile {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-tile .label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
.metric-tile .value { font-family: 'Space Mono', monospace; font-size: 1.6rem; color: #93c5fd; }

/* Factor rows */
.factor-row {
    display: flex; align-items: center;
    background: #1e293b;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.4rem;
    border-left: 3px solid;
    font-size: 0.85rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #1d4ed8, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    width: 100%;
    transition: all 0.2s !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #1e40af, #1d4ed8) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.4) !important;
}

/* Section headings */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.75rem;
}

.divider { border-top: 1px solid #1e293b; margin: 1.5rem 0; }

/* Tab overrides */
.stTabs [data-baseweb="tab-list"] { background: #0f172a; border-radius: 8px; }
.stTabs [data-baseweb="tab"] { color: #64748b !important; font-size: 0.85rem; }
.stTabs [aria-selected="true"] { color: #93c5fd !important; }

/* Selectbox / slider labels */
label { color: #94a3b8 !important; font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Load / train model ────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "flight_delay_model.pkl")

@st.cache_resource(show_spinner="Training models on first run…")
def load_or_train():
    if not os.path.exists(MODEL_PATH):
        import model_training
        model_training.run()
    return joblib.load(MODEL_PATH)

bundle = load_or_train()
models      = bundle["models"]
best_model  = bundle["best_model"]
encoders    = bundle["encoders"]
scaler      = bundle["scaler"]
metrics     = bundle["metrics"]

# ── Helper to draw matplotlib figures in dark theme ──────────────────────────
DARK_BG  = "#0f172a"
GRID_COL = "#1e293b"
TEXT_COL = "#94a3b8"
BLUE     = "#3b82f6"
GREEN    = "#22c55e"
RED      = "#ef4444"
AMBER    = "#f59e0b"

def dark_fig(w=8, h=4):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COL)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(color=GRID_COL, linewidth=0.6)
    ax.title.set_color("#e2e8f0")
    return fig, ax

# ── Load dataset for dashboard ────────────────────────────────────────────────
@st.cache_data
def load_data():
    from utils import engineer_features
    df = pd.read_csv(os.path.join(BASE_DIR, "dataset.csv"))
    df = engineer_features(df)
    return df

df_raw = load_data()

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — inputs
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="section-label">✈ Flight Parameters</div>', unsafe_allow_html=True)

    airline = st.selectbox(
        "Airline",
        options=list(AIRLINE_NAMES.keys()),
        format_func=lambda k: f"{k} — {AIRLINE_NAMES[k]}",
    )

    airport_codes = list(AIRPORTS.keys())
    origin = st.selectbox(
        "Origin Airport",
        options=airport_codes,
        format_func=lambda k: f"{k} · {AIRPORTS[k]}",
    )
    destination = st.selectbox(
        "Destination Airport",
        options=airport_codes,
        index=1,
        format_func=lambda k: f"{k} · {AIRPORTS[k]}",
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">🕐 Schedule</div>', unsafe_allow_html=True)

    dep_hour   = st.slider("Departure Hour (24h)", 0, 23, 8)
    dep_minute = st.select_slider("Departure Minute", options=[0, 15, 30, 45], value=0)
    dep_time   = dep_hour * 100 + dep_minute

    day_of_week = st.select_slider(
        "Day of Week",
        options=list(DAY_NAMES.keys()),
        format_func=lambda d: DAY_NAMES[d],
        value=1,
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📏 Route & Weather</div>', unsafe_allow_html=True)

    distance = st.slider("Distance (miles)", 50, 3000, 800, step=50)

    weather = st.selectbox("Weather Condition", WEATHER_CONDITIONS)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">🤖 Model</div>', unsafe_allow_html=True)

    selected_model = st.selectbox("Prediction Model", list(models.keys()), index=list(models.keys()).index(best_model))

    predict_btn = st.button("🔍 Predict Delay")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
  <h1>✈ Flight Delay Prediction System</h1>
  <p>ML-powered delay risk analysis · Logistic Regression · Random Forest · Gradient Boosting</p>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_dashboard, tab_model = st.tabs(["🔍 Predict", "📊 Dashboard", "🧠 Model Insights"])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Predict
# ══════════════════════════════════════════════════════════════════════════════

with tab_predict:

    if predict_btn:
        with st.spinner("Running prediction…"):
            features = preprocess_input(
                airline, origin, destination, dep_time,
                distance, day_of_week, weather, encoders, scaler
            )
            model   = models[selected_model]
            prob    = model.predict_proba(features)[0][1]
            delayed = prob >= 0.5
            interp  = get_delay_interpretation(prob)

        # ── Result banner ─────────────────────────────────────────────────────
        border_col = interp["color"]
        bg_alpha   = "20"
        result_label = "DELAYED" if delayed else "ON TIME"
        st.markdown(f"""
        <div class="result-card" style="background:{border_col}{bg_alpha}; border-color:{border_col};">
          <h2 style="color:{border_col};">{interp['emoji']} Flight Predicted: {result_label}</h2>
          <p>{interp['advice']}</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Metric row ────────────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        def metric_html(label, value):
            return f"""
            <div class="metric-tile">
              <div class="label">{label}</div>
              <div class="value">{value}</div>
            </div>"""

        m1.markdown(metric_html("Delay Probability", f"{prob*100:.1f}%"), unsafe_allow_html=True)
        m2.markdown(metric_html("Risk Level", interp["level"].split()[0]), unsafe_allow_html=True)
        m3.markdown(metric_html("Model", selected_model.split()[0]), unsafe_allow_html=True)
        m4.markdown(metric_html("Route", f"{origin}→{destination}"), unsafe_allow_html=True)

        st.markdown("")

        col_a, col_b = st.columns([1, 1])

        # ── Gauge chart ───────────────────────────────────────────────────────
        with col_a:
            st.markdown('<div class="section-label">Delay Probability Gauge</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 3), subplot_kw=dict(polar=True), facecolor=DARK_BG)
            ax.set_facecolor(DARK_BG)

            theta_max = np.pi
            n = 200
            thetas = np.linspace(0, theta_max, n)

            # colour segments
            for i, (start, end, color) in enumerate([
                (0, 0.25, "#22c55e"),
                (0.25, 0.50, "#84cc16"),
                (0.50, 0.75, "#f59e0b"),
                (0.75, 1.00, "#ef4444"),
            ]):
                t = np.linspace(start * np.pi, end * np.pi, 50)
                ax.bar(t, [0.35]*50, width=np.pi/n*3, bottom=0.65,
                       color=color, alpha=0.85, linewidth=0)

            # needle
            needle_angle = prob * np.pi
            ax.annotate("", xy=(needle_angle, 1.0), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="white", lw=2))

            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
            ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color=TEXT_COL, fontsize=8)
            ax.spines["polar"].set_visible(False)
            ax.set_theta_offset(0)
            ax.set_theta_direction(1)

            # limit to upper half
            ax.set_thetamin(0); ax.set_thetamax(180)

            ax.text(np.pi/2, 0.35, f"{prob*100:.1f}%",
                    ha="center", va="center", color="white",
                    fontsize=18, fontweight="bold",
                    fontfamily="monospace")

            fig.patch.set_facecolor(DARK_BG)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # ── Contributing factors ──────────────────────────────────────────────
        with col_b:
            st.markdown('<div class="section-label">Contributing Factors</div>', unsafe_allow_html=True)
            factors = simulate_realtime_factors(airline, origin, weather, day_of_week, dep_hour)
            for name, score, direction in factors:
                bar_w = int(score * 100)
                col_map = {"negative": RED, "positive": GREEN, "neutral": AMBER}
                icon_map = {"negative": "↑", "positive": "↓", "neutral": "→"}
                c = col_map[direction]
                i = icon_map[direction]
                st.markdown(f"""
                <div class="factor-row" style="border-left-color:{c};">
                  <span style="flex:1;color:#e2e8f0;">{name}</span>
                  <span style="width:80px;background:#0f172a;border-radius:4px;height:8px;overflow:hidden;margin:0 0.75rem;">
                    <span style="display:block;height:100%;width:{bar_w}%;background:{c};border-radius:4px;"></span>
                  </span>
                  <span style="color:{c};font-family:monospace;width:3rem;text-align:right;">{score:.0%} {i}</span>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── All-model comparison ──────────────────────────────────────────────
        st.markdown('<div class="section-label">All-Model Comparison</div>', unsafe_allow_html=True)
        comp_data = {}
        for mname, mobj in models.items():
            f2 = preprocess_input(airline, origin, destination, dep_time,
                                  distance, day_of_week, weather, encoders, scaler)
            p = mobj.predict_proba(f2)[0][1]
            comp_data[mname] = p

        fig2, ax2 = dark_fig(8, 3)
        names  = list(comp_data.keys())
        probs2 = list(comp_data.values())
        colors = [GREEN if p < 0.5 else RED for p in probs2]
        bars   = ax2.barh(names, [p * 100 for p in probs2], color=colors, height=0.5)
        ax2.axvline(50, color=AMBER, linestyle="--", lw=1.2, label="50% threshold")
        ax2.set_xlim(0, 100)
        ax2.set_xlabel("Delay Probability (%)")
        ax2.set_title("Prediction Comparison Across Models")
        ax2.legend(facecolor=DARK_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
        for bar, p in zip(bars, probs2):
            ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                     f"{p*100:.1f}%", va="center", color=TEXT_COL, fontsize=9)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    else:
        # ── Placeholder hint ──────────────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center; padding:4rem 2rem; color:#334155;">
          <div style="font-size:4rem; margin-bottom:1rem;">✈️</div>
          <div style="font-family:'Space Mono',monospace; font-size:1.1rem; color:#475569;">
            Configure your flight in the sidebar<br>and press <span style="color:#3b82f6;">Predict Delay</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Dashboard
# ══════════════════════════════════════════════════════════════════════════════

with tab_dashboard:
    st.markdown('<div class="section-label">Historical Delay Analytics</div>', unsafe_allow_html=True)

    row1c1, row1c2 = st.columns(2)

    # ── Airline delay rate ────────────────────────────────────────────────────
    with row1c1:
        airline_stats = df_raw.groupby("Airline")["Delayed"].mean().sort_values(ascending=True)
        fig, ax = dark_fig(6, 4)
        colors  = [RED if v > 0.5 else GREEN if v < 0.3 else AMBER for v in airline_stats.values]
        ax.barh(airline_stats.index, airline_stats.values * 100, color=colors, height=0.6)
        ax.set_xlabel("Delay Rate (%)")
        ax.set_title("Delay Rate by Airline")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.axvline(50, color=AMBER, linestyle="--", lw=1, alpha=0.6)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Weekly delay trend ────────────────────────────────────────────────────
    with row1c2:
        weekly = df_raw.groupby("DayOfWeek")["Delayed"].mean()
        fig, ax = dark_fig(6, 4)
        ax.plot(list(DAY_NAMES.values()), weekly.values * 100,
                marker="o", color=BLUE, linewidth=2, markersize=7, markerfacecolor=GREEN)
        ax.fill_between(range(len(weekly)), weekly.values * 100, alpha=0.1, color=BLUE)
        ax.set_xticks(range(len(weekly)))
        ax.set_xticklabels(list(DAY_NAMES.values()), rotation=30, ha="right")
        ax.set_ylabel("Delay Rate (%)")
        ax.set_title("Weekly Delay Pattern")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    row2c1, row2c2 = st.columns(2)

    # ── Airport delay heatmap ─────────────────────────────────────────────────
    with row2c1:
        orig_stats = df_raw.groupby("Origin")["Delayed"].mean().sort_values(ascending=False).head(8)
        fig, ax = dark_fig(6, 4)
        bars = ax.bar(orig_stats.index, orig_stats.values * 100,
                      color=[RED if v > 0.5 else AMBER if v > 0.3 else GREEN for v in orig_stats.values])
        ax.set_ylabel("Delay Rate (%)")
        ax.set_title("Top Origin Airports by Delay Rate")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{bar.get_height():.0f}%", ha="center", va="bottom", color=TEXT_COL, fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Weather delay rates ───────────────────────────────────────────────────
    with row2c2:
        wx_stats = df_raw.groupby("WeatherCondition")["Delayed"].mean().sort_values()
        fig, ax  = dark_fig(6, 4)
        wx_colors = [WEATHER_DELAY_RISK.get(w, 0.2) for w in wx_stats.index]
        wx_norm   = [(c - min(wx_colors)) / (max(wx_colors) - min(wx_colors) + 1e-9) for c in wx_colors]
        cmap      = plt.cm.RdYlGn_r
        bar_cols  = [cmap(v) for v in wx_norm]
        ax.barh(wx_stats.index, wx_stats.values * 100, color=bar_cols, height=0.55)
        ax.set_xlabel("Delay Rate (%)")
        ax.set_title("Delay Rate by Weather Condition")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Distance distribution ─────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Distance vs. Delay</div>', unsafe_allow_html=True)
    fig, ax = dark_fig(12, 3.5)
    on_time = df_raw[df_raw["Delayed"] == 0]["Distance"]
    delayed = df_raw[df_raw["Delayed"] == 1]["Distance"]
    ax.hist(on_time, bins=25, color=GREEN, alpha=0.6, label="On-Time", density=True)
    ax.hist(delayed, bins=25, color=RED,   alpha=0.6, label="Delayed",  density=True)
    ax.set_xlabel("Distance (miles)")
    ax.set_ylabel("Density")
    ax.set_title("Flight Distance Distribution: On-Time vs. Delayed")
    ax.legend(facecolor=DARK_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — Model Insights
# ══════════════════════════════════════════════════════════════════════════════

with tab_model:
    st.markdown('<div class="section-label">Model Performance Summary</div>', unsafe_allow_html=True)

    # ── Accuracy / AUC table ──────────────────────────────────────────────────
    perf_rows = []
    for mname, mmet in metrics.items():
        perf_rows.append({
            "Model":        mname,
            "Accuracy":     f"{mmet['accuracy']*100:.1f}%",
            "AUC-ROC":      f"{mmet['auc']:.3f}",
            "CV Mean":      f"{mmet['cv_mean']*100:.1f}%",
            "CV Std":       f"±{mmet['cv_std']*100:.1f}%",
            "Best ★":       "★" if mname == best_model else "",
        })
    st.dataframe(pd.DataFrame(perf_rows).set_index("Model"), use_container_width=True)

    col_m1, col_m2 = st.columns(2)

    # ── Accuracy bars ─────────────────────────────────────────────────────────
    with col_m1:
        fig, ax = dark_fig(6, 3.5)
        mnames = list(metrics.keys())
        accs   = [metrics[m]["accuracy"] * 100 for m in mnames]
        aucs   = [metrics[m]["auc"] * 100       for m in mnames]
        x      = np.arange(len(mnames))
        ax.bar(x - 0.2, accs, 0.35, label="Accuracy", color=BLUE,  alpha=0.85)
        ax.bar(x + 0.2, aucs, 0.35, label="AUC-ROC",  color=GREEN, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ", "\n") for m in mnames])
        ax.set_ylim(0, 110)
        ax.set_ylabel("Score (%)")
        ax.set_title("Accuracy vs AUC-ROC")
        ax.legend(facecolor=DARK_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Feature importance (Random Forest) ───────────────────────────────────
    with col_m2:
        if "Random Forest" in models:
            rf  = models["Random Forest"]
            imp = rf.feature_importances_
            fc  = bundle["feature_cols"]
            idx = np.argsort(imp)[::-1]
            fig, ax = dark_fig(6, 3.5)
            colors  = [BLUE if i < 5 else "#475569" for i in range(len(imp))]
            ax.bar(range(len(imp)), imp[idx] * 100,
                   color=[colors[i] for i in range(len(imp))])
            ax.set_xticks(range(len(imp)))
            ax.set_xticklabels([fc[i] for i in idx], rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Importance (%)")
            ax.set_title("Feature Importance — Random Forest")
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ── Confusion matrix for selected model ───────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Confusion Matrix</div>', unsafe_allow_html=True)

    sel = st.selectbox("Select model for confusion matrix", list(models.keys()), key="cm_sel")
    cm  = metrics[sel]["confusion"]

    fig, ax = plt.subplots(figsize=(5, 4), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["On-Time", "Delayed"],
                yticklabels=["On-Time", "Delayed"],
                ax=ax, linewidths=0.5, linecolor=DARK_BG,
                annot_kws={"size": 14, "color": "white"})
    ax.set_title(f"Confusion Matrix — {sel}", color="#e2e8f0", pad=12)
    ax.set_xlabel("Predicted", color=TEXT_COL)
    ax.set_ylabel("Actual",    color=TEXT_COL)
    ax.tick_params(colors=TEXT_COL)
    fig.tight_layout()
    col_cm, _ = st.columns([0.5, 0.5])
    with col_cm:
        st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Classification report ─────────────────────────────────────────────────
    st.markdown('<div class="section-label">Classification Report</div>', unsafe_allow_html=True)
    rep = metrics[sel]["report"]
    rep_df = pd.DataFrame({
        "Class":     ["On-Time (0)", "Delayed (1)", "Macro Avg", "Weighted Avg"],
        "Precision": [rep["0"]["precision"], rep["1"]["precision"],
                      rep["macro avg"]["precision"], rep["weighted avg"]["precision"]],
        "Recall":    [rep["0"]["recall"], rep["1"]["recall"],
                      rep["macro avg"]["recall"], rep["weighted avg"]["recall"]],
        "F1-Score":  [rep["0"]["f1-score"], rep["1"]["f1-score"],
                      rep["macro avg"]["f1-score"], rep["weighted avg"]["f1-score"]],
    })
    rep_df[["Precision", "Recall", "F1-Score"]] = (
        rep_df[["Precision", "Recall", "F1-Score"]].applymap(lambda v: f"{v:.3f}")
    )
    st.dataframe(rep_df.set_index("Class"), use_container_width=True)
