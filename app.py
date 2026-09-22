"""
app.py  ─  Breast Cancer Diagnosis: Interactive Data Analytics & Predictive Modeling

Run:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st

from utils import (
    load_data, get_feature_cols, load_pipeline, load_metrics,
    load_rf_importances, models_trained, PALETTE, SNS_PALETTE,
    MODEL_LABELS, make_fig, plot_confusion_matrix, plot_roc_curves,
)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Breast Cancer Diagnosis | Data Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── font & base ── */
html, body, [class*="css"] { font-family: 'Segoe UI', system-ui, sans-serif; }

/* ── sidebar shell ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1A3E 0%, #112252 55%, #1A3A6B 100%);
    border-right: 1px solid rgba(255,255,255,0.07);
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}
[data-testid="stSidebar"] * { color: white !important; }

/* ── hide default radio dots ── */
[data-testid="stSidebar"] .stRadio > div { gap: 2px !important; }
[data-testid="stSidebar"] .stRadio [role="radiogroup"] { gap: 2px !important; }
[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center !important;
    padding: 0.55rem 1.1rem !important;
    margin: 1px 0 !important;
    border-radius: 8px !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: background 0.15s ease !important;
    letter-spacing: 0.01em !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.09) !important;
}
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:has(input:checked),
[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: rgba(59,130,246,0.28) !important;
    border-left: 3px solid #60A5FA !important;
    font-weight: 600 !important;
}
/* hide the actual radio circle */
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > div:first-child {
    display: none !important;
}

/* ── KPI card ── */
.kpi-card {
    background: linear-gradient(135deg, #1A3A6B 0%, #2563EB 100%);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    color: white;
    text-align: center;
    box-shadow: 0 4px 18px rgba(37,99,235,0.18);
}
.kpi-label  { font-size: 0.82rem; opacity: .78; text-transform: uppercase; letter-spacing: .06em; }
.kpi-value  { font-size: 2.2rem; font-weight: 700; line-height: 1.15; }
.kpi-sub    { font-size: 0.80rem; opacity: .70; }

/* ── benign / malignant cards ── */
.kpi-b { background: linear-gradient(135deg, #1D4ED8, #3B82F6); }
.kpi-m { background: linear-gradient(135deg, #991B1B, #EF4444); }

/* ── section header ── */
.section-header {
    font-size: 1.45rem; font-weight: 700; color: #0F2044;
    border-left: 4px solid #2563EB; padding-left: .7rem;
    margin-bottom: 1rem;
}
/* ── metric pill ── */
.metric-pill {
    display: inline-block; background: #EFF6FF; color: #1E40AF;
    border-radius: 20px; padding: .25rem .75rem; font-size: .85rem;
    font-weight: 600; margin: .2rem;
}
/* ── disclaimer ── */
.disclaimer {
    background: #FFF7ED; border-left: 4px solid #F97316;
    border-radius: 8px; padding: .85rem 1.1rem;
    font-size: .86rem; color: #7C2D12;
}
/* ── workflow step ── */
.wf-step {
    background: #EFF6FF; border-radius: 10px; text-align: center;
    padding: .6rem .5rem; font-size: .82rem; font-weight: 600; color: #1E40AF;
}
.wf-arrow { text-align: center; font-size: 1.1rem; color: #94A3B8; margin: 0; }

/* ── tech badge ── */
.tech-badge {
    display:inline-block; background:#F0F9FF; border:1px solid #BAE6FD;
    border-radius:8px; padding:.35rem .7rem; font-size:.82rem;
    font-weight:600; color:#0369A1; margin:.2rem;
}
/* ── model card ── */
.model-card {
    border: 1px solid #E5E7EB; border-radius: 14px;
    padding: 1.1rem 1.3rem; background: #F8FAFF;
    box-shadow: 0 2px 10px rgba(0,0,0,.05);
}
.model-card h4 { margin: 0 0 .6rem 0; color: #0F2044; font-size: 1rem; }

/* ── prediction result cards ── */
.pred-benign {
    background: linear-gradient(135deg,#1D4ED8,#3B82F6); color:white;
    border-radius:16px; padding:1.6rem 1.8rem; text-align:center;
    font-size:2rem; font-weight:700; box-shadow:0 4px 20px rgba(37,99,235,.25);
}
.pred-malignant {
    background: linear-gradient(135deg,#991B1B,#EF4444); color:white;
    border-radius:16px; padding:1.6rem 1.8rem; text-align:center;
    font-size:2rem; font-weight:700; box-shadow:0 4px 20px rgba(220,38,38,.25);
}
.pred-label { font-size:.9rem; opacity:.85; margin-top:.3rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Logo / title area ────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding: 1.4rem 1.2rem 1rem 1.2rem;">
        <div style="display:flex; align-items:center; gap:.6rem; margin-bottom:.35rem;">
            <span style="font-size:1.7rem; line-height:1;">🩺</span>
            <div>
                <div style="font-size:1.05rem; font-weight:700; line-height:1.2;
                            letter-spacing:.01em;">Breast Cancer</div>
                <div style="font-size:0.75rem; opacity:.65; font-weight:400;
                            letter-spacing:.04em; text-transform:uppercase;">
                    Diagnostic Analytics
                </div>
            </div>
        </div>
        <div style="height:1px; background:rgba(255,255,255,0.12);
                    margin-top:.75rem;"></div>
        <div style="font-size:.7rem; opacity:.45; margin-top:.5rem;
                    letter-spacing:.05em; text-transform:uppercase;">
            Navigation
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Navigation ───────────────────────────────────────────────────────────
    page = st.radio(
        "Navigate",
        ["🏠  Home",
         "📊  Dataset Explorer",
         "📈  Exploratory Analysis",
         "🔬  Feature Insights",
         "🤖  Model Performance",
         "🔮  Prediction",
         "ℹ️  About Project"],
        label_visibility="collapsed",
    )

# ─── pre-load default data and models ─────────────────────────────────────────
_default_df   = load_data()
_default_fcols = get_feature_cols()
metrics        = load_metrics()
rf_imp         = load_rf_importances()


# ─── CSV upload validation helper ─────────────────────────────────────────────
def validate_uploaded_csv(raw_df: pd.DataFrame):
    """
    Returns (cleaned_df, feature_cols, error_message).
    error_message is None on success, a string on failure.
    """
    # Drop the all-null artifact column if present
    if "Unnamed: 32" in raw_df.columns:
        raw_df = raw_df.drop(columns=["Unnamed: 32"])

    # Must have a diagnosis column
    if "diagnosis" not in raw_df.columns:
        return None, None, (
            "The uploaded CSV does not contain a **'diagnosis'** column. "
            "This app requires a column named `diagnosis` with values `M` (Malignant) "
            "and `B` (Benign). Please upload the correct file."
        )

    # diagnosis values must include at least one of B / M
    valid_vals = set(raw_df["diagnosis"].dropna().unique())
    if not valid_vals.issubset({"M", "B"}):
        unexpected = valid_vals - {"M", "B"}
        return None, None, (
            f"Unexpected values in 'diagnosis' column: **{unexpected}**. "
            "Expected only `M` (Malignant) and `B` (Benign)."
        )
    if not {"M", "B"} & valid_vals:
        return None, None, (
            "The 'diagnosis' column contains no valid `M` or `B` values."
        )

    # Derive numeric feature columns (exclude id, diagnosis, diagnosis_label)
    exclude = {"id", "diagnosis", "diagnosis_label", "diagnosis_encoded"}
    fcols = [c for c in raw_df.columns
             if c not in exclude and pd.api.types.is_numeric_dtype(raw_df[c])]

    if len(fcols) == 0:
        return None, None, (
            "No numeric feature columns found in the uploaded CSV. "
            "The file must contain numeric measurement columns."
        )

    # Add helper columns expected by the rest of the app
    raw_df = raw_df.copy()
    raw_df["diagnosis_label"] = raw_df["diagnosis"].map({"B": "Benign", "M": "Malignant"})

    return raw_df, fcols, None


# ─── Active dataset: uploaded file wins over default ──────────────────────────
def get_active_dataset():
    """Return (df, feature_cols) — either from upload or default."""
    if st.session_state.get("uploaded_df") is not None:
        return st.session_state["uploaded_df"], st.session_state["uploaded_fcols"]
    return _default_df, _default_fcols


df, feature_cols = get_active_dataset()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def kpi(label, value, sub="", extra_cls=""):
    return (f'<div class="kpi-card {extra_cls}">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>')

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def disclaimer():
    st.markdown(
        '<div class="disclaimer">⚕️ <strong>Educational Demonstration Only:</strong> '
        'This application is intended for academic and educational purposes. '
        'It is <strong>not</strong> a medical diagnostic tool and must not be used as a '
        'substitute for professional medical advice, examination, or diagnosis.</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":

    # Hero
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0F2044 0%,#1A3A6B 50%,#2563EB 100%);
                border-radius:18px;padding:2.2rem 2.4rem;margin-bottom:1.6rem;color:white;'>
        <h1 style='margin:0;font-size:2.1rem;font-weight:800;'>
            🩺 Breast Cancer Diagnosis
        </h1>
        <p style='margin:.5rem 0 0;font-size:1.1rem;opacity:.85;'>
            Interactive Data Analytics &amp; Predictive Modeling
        </p>
        <p style='margin:.4rem 0 0;font-size:.88rem;opacity:.65;'>
            Breast Cancer Wisconsin (Diagnostic) Dataset &nbsp;·&nbsp;
            Exploratory Data Analysis &amp; Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI cards ──────────────────────────────────────────────────────────────
    n_total    = len(df)
    n_features = len(feature_cols)
    n_benign   = (df["diagnosis"] == "B").sum()
    n_malig    = (df["diagnosis"] == "M").sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi("Total Records", n_total, "patient samples"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("Numeric Features", n_features, "30 diagnostic measurements"), unsafe_allow_html=True)
    with c3:
        pct_b = f"{n_benign/n_total*100:.1f}% of dataset"
        st.markdown(kpi("Benign Cases", n_benign, pct_b, "kpi-b"), unsafe_allow_html=True)
    with c4:
        pct_m = f"{n_malig/n_total*100:.1f}% of dataset"
        st.markdown(kpi("Malignant Cases", n_malig, pct_m, "kpi-m"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Diagnosis distribution + workflow ─────────────────────────────────────
    col_chart, col_wf = st.columns([1.4, 1])

    with col_chart:
        section("📊 Diagnosis Distribution")
        fig, ax = make_fig(figsize=(6, 3.5))
        counts = df["diagnosis_label"].value_counts()
        bars = ax.bar(counts.index, counts.values,
                      color=[PALETTE["Benign"], PALETTE["Malignant"]],
                      edgecolor="white", width=0.45, zorder=3)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 4, str(int(bar.get_height())),
                    ha="center", va="bottom", fontweight="bold", fontsize=12)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title("Benign vs Malignant — Class Distribution", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.4)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_wf:
        section("⚙️ Project Workflow")
        steps = ["📂 Data Loading", "🧹 Cleaning",
                 "🔍 Exploration", "📐 Feature Analysis",
                 "⚙️ Preprocessing", "🤖 ML Modeling",
                 "📊 Evaluation", "🔮 Prediction"]
        for i, s in enumerate(steps):
            st.markdown(f'<div class="wf-step">{s}</div>', unsafe_allow_html=True)
            if i < len(steps) - 1:
                st.markdown('<div class="wf-arrow">↓</div>', unsafe_allow_html=True)

    # ── Technologies ──────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section("🛠️ Technologies Used")
    techs = ["Python 3", "Pandas", "NumPy", "Matplotlib",
             "Seaborn", "Scikit-learn", "Streamlit", "Joblib", "Jupyter"]
    badge_html = " ".join(f'<span class="tech-badge">{t}</span>' for t in techs)
    st.markdown(badge_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    disclaimer()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATASET EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Dataset Explorer":
    section("📊 Dataset Explorer")

    # ── CSV Upload ────────────────────────────────────────────────────────────
    with st.expander("📂 Upload a CSV dataset (optional)", expanded=False):
        st.markdown(
            "Upload any CSV that has a **`diagnosis`** column (`M`/`B`) "
            "and numeric feature columns. When uploaded, all analysis on this "
            "page and the Exploratory Analysis page will use your file. "
            "Close or clear the upload to return to the default `data.csv`."
        )
        uploaded_file = st.file_uploader(
            "Choose a CSV file", type=["csv"], key="csv_uploader"
        )

        if uploaded_file is not None:
            try:
                raw = pd.read_csv(uploaded_file)
                validated_df, validated_fcols, err = validate_uploaded_csv(raw)
                if err:
                    st.error(f"**Validation error:** {err}")
                    # Clear any previous valid upload
                    st.session_state["uploaded_df"]    = None
                    st.session_state["uploaded_fcols"] = None
                else:
                    st.session_state["uploaded_df"]    = validated_df
                    st.session_state["uploaded_fcols"] = validated_fcols
                    st.success(
                        f"CSV loaded: **{uploaded_file.name}** — "
                        f"{validated_df.shape[0]} rows, "
                        f"{len(validated_fcols)} numeric features."
                    )
            except Exception as exc:
                st.error(f"Could not read the file: {exc}")
                st.session_state["uploaded_df"]    = None
                st.session_state["uploaded_fcols"] = None
        else:
            # File uploader cleared — fall back to default
            st.session_state["uploaded_df"]    = None
            st.session_state["uploaded_fcols"] = None

    # Re-resolve active dataset after upload widget runs
    df, feature_cols = get_active_dataset()

    is_uploaded = st.session_state.get("uploaded_df") is not None
    if is_uploaded:
        st.info("📂 Showing analysis for the **uploaded CSV**. Clear the upload to revert to `data.csv`.")

    # Filter
    filt = st.radio("Filter by diagnosis:", ["All", "Benign", "Malignant"],
                    horizontal=True)
    if filt == "Benign":
        view = df[df["diagnosis"] == "B"]
    elif filt == "Malignant":
        view = df[df["diagnosis"] == "M"]
    else:
        view = df

    # Dimensions
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", view.shape[0])
    c2.metric("Feature Columns", len(feature_cols))
    c3.metric("Benign",    (view["diagnosis"]=="B").sum())
    c4.metric("Malignant", (view["diagnosis"]=="M").sum())

    st.markdown("---")

    tab_prev, tab_stats, tab_qual = st.tabs([
        "📋 Data Preview", "📐 Statistics", "🔎 Data Quality"
    ])

    with tab_prev:
        n_feat_preview = min(8, len(feature_cols))
        st.markdown(f"**First 10 rows** (showing diagnosis + first {n_feat_preview} feature columns):")
        display_cols = ["diagnosis"] + feature_cols[:n_feat_preview]
        st.dataframe(view[display_cols].head(10), use_container_width=True)
        with st.expander(f"Show all {len(feature_cols)} feature columns"):
            st.dataframe(view[["diagnosis"] + feature_cols], use_container_width=True)

    with tab_stats:
        st.markdown("**Descriptive statistics for numerical features** (filtered view):")
        stats = view[feature_cols].describe().T.round(4)
        st.dataframe(stats, use_container_width=True)

    with tab_qual:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Missing values per column:**")
            miss = df[feature_cols].isnull().sum()
            miss_df = miss[miss > 0].reset_index()
            miss_df.columns = ["Column", "Missing Count"]
            if miss_df.empty:
                st.success("✅ No missing values in any feature column.")
            else:
                st.dataframe(miss_df, use_container_width=True)

        with col_b:
            st.markdown("**Data types:**")
            dtype_df = pd.DataFrame({
                "Column": feature_cols,
                "Type":   [str(df[c].dtype) for c in feature_cols],
            })
            st.dataframe(dtype_df, use_container_width=True)

        dupes = df.duplicated().sum()
        if dupes == 0:
            st.success("✅ No duplicate rows found.")
        else:
            st.warning(f"⚠️ {dupes} duplicate rows detected.")

        st.markdown("---")
        st.markdown("**Target class distribution:**")
        dist = df["diagnosis_label"].value_counts().reset_index()
        dist.columns = ["Diagnosis", "Count"]
        dist["Percentage"] = (dist["Count"] / len(df) * 100).round(1).astype(str) + "%"
        st.dataframe(dist, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EXPLORATORY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Exploratory Analysis":
    section("📈 Exploratory Data Analysis")

    # Re-resolve in case user uploaded on Dataset Explorer page this session
    df, feature_cols = get_active_dataset()
    if st.session_state.get("uploaded_df") is not None:
        st.info("📂 Showing analysis for the **uploaded CSV**.")

    tab_dist, tab_comp, tab_corr, tab_rel = st.tabs([
        "📊 Distribution", "⚖️ Comparison", "🔥 Correlation", "🔗 Relationships"
    ])

    # ── Tab: Distribution ─────────────────────────────────────────────────────
    with tab_dist:
        feat = st.selectbox("Select a feature:", feature_cols, index=0)
        data = df[feat]

        m_col, sd_col, mn_col, mx_col = st.columns(4)
        m_col.metric("Mean",   f"{data.mean():.4f}")
        sd_col.metric("Std Dev", f"{data.std():.4f}")
        mn_col.metric("Min",   f"{data.min():.4f}")
        mx_col.metric("Max",   f"{data.max():.4f}")
        st.metric("Median", f"{data.median():.4f}")

        fig, axes = make_fig(1, 2, figsize=(12, 4))
        # histogram
        axes[0].hist(data, bins=30, color="#3B82F6", edgecolor="white", alpha=0.85)
        axes[0].axvline(data.mean(),   color="#EF4444", lw=2, linestyle="--", label=f"Mean={data.mean():.2f}")
        axes[0].axvline(data.median(), color="#F59E0B", lw=2, linestyle=":",  label=f"Median={data.median():.2f}")
        axes[0].set_title(f"Histogram — {feat}", fontsize=11, fontweight="bold")
        axes[0].set_xlabel(feat); axes[0].set_ylabel("Count")
        axes[0].legend(fontsize=9); axes[0].grid(axis="y", alpha=0.4)

        # KDE by class
        for label, color in [("Benign", "#3B82F6"), ("Malignant", "#EF4444")]:
            subset = df[df["diagnosis_label"] == label][feat]
            sns.kdeplot(subset, ax=axes[1], label=label, color=color,
                        fill=True, alpha=0.30, linewidth=2)
        axes[1].set_title(f"KDE by Diagnosis — {feat}", fontsize=11, fontweight="bold")
        axes[1].set_xlabel(feat); axes[1].legend(fontsize=9)
        axes[1].grid(axis="y", alpha=0.4)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Tab: Comparison ───────────────────────────────────────────────────────
    with tab_comp:
        comp_feats = st.multiselect(
            "Select features to compare (up to 6):",
            options=feature_cols,
            default=["radius_mean", "area_mean", "concavity_mean",
                     "concave points_mean", "perimeter_mean", "texture_mean"],
            max_selections=6,
        )
        if not comp_feats:
            st.info("Select at least one feature above.")
        else:
            n = len(comp_feats)
            ncols = min(3, n)
            nrows = (n + ncols - 1) // ncols
            fig, axes = make_fig(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
            if n == 1:
                axes = [axes]
            elif nrows == 1:
                axes = list(axes)
            else:
                axes = [ax for row in axes for ax in row]

            for i, feat in enumerate(comp_feats):
                sns.boxplot(data=df, x="diagnosis_label", y=feat, ax=axes[i],
                            palette={"Benign": "#3B82F6", "Malignant": "#EF4444"},
                            width=0.45)
                axes[i].set_title(feat.replace("_", " ").title(),
                                  fontsize=10, fontweight="bold")
                axes[i].set_xlabel("")

            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Group statistics table
            st.markdown("**Group Statistics (mean ± std):**")
            rows = []
            for feat in comp_feats:
                b_vals = df[df["diagnosis"] == "B"][feat]
                m_vals = df[df["diagnosis"] == "M"][feat]
                rows.append({
                    "Feature":            feat,
                    "Benign mean":        round(b_vals.mean(), 4),
                    "Benign std":         round(b_vals.std(),  4),
                    "Malignant mean":     round(m_vals.mean(), 4),
                    "Malignant std":      round(m_vals.std(),  4),
                    "Δ mean (M−B)":       round(m_vals.mean() - b_vals.mean(), 4),
                    "% diff":             f"{(m_vals.mean()-b_vals.mean())/b_vals.mean()*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ── Tab: Correlation ──────────────────────────────────────────────────────
    with tab_corr:
        st.markdown("**Pearson correlation matrix** for all 30 numeric features:")
        corr = df[feature_cols].corr()
        fig, ax = make_fig(figsize=(16, 12))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                    vmin=-1, vmax=1, annot=False,
                    linewidths=0.3, ax=ax,
                    cbar_kws={"shrink": 0.65})
        ax.set_title("Correlation Heatmap — All 30 Features",
                     fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Top highly-correlated pairs
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                r = corr.iloc[i, j]
                if abs(r) > 0.85:
                    pairs.append({
                        "Feature 1":   corr.columns[i],
                        "Feature 2":   corr.columns[j],
                        "Correlation": round(r, 3),
                    })
        if pairs:
            top_pairs = (pd.DataFrame(pairs)
                         .sort_values("Correlation", ascending=False)
                         .head(15))
            with st.expander(f"Top highly-correlated pairs (|r| > 0.85) — {len(pairs)} found"):
                st.dataframe(top_pairs, use_container_width=True)

    # ── Tab: Relationships ────────────────────────────────────────────────────
    with tab_rel:
        col_x, col_y = st.columns(2)
        x_feat = col_x.selectbox("X-axis feature:", feature_cols, index=0)
        y_feat = col_y.selectbox("Y-axis feature:", feature_cols, index=2)

        fig, ax = make_fig(figsize=(8, 5))
        for label, color, marker in [("Benign", "#3B82F6", "o"),
                                      ("Malignant", "#EF4444", "^")]:
            subset = df[df["diagnosis_label"] == label]
            ax.scatter(subset[x_feat], subset[y_feat],
                       c=color, marker=marker, alpha=0.55, s=28, label=label)
        ax.set_xlabel(x_feat.replace("_", " ").title(), fontsize=10)
        ax.set_ylabel(y_feat.replace("_", " ").title(), fontsize=10)
        ax.set_title(f"{x_feat} vs {y_feat}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.35)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # correlation coefficient
        r = df[[x_feat, y_feat]].corr().iloc[0, 1]
        st.info(f"Pearson r between **{x_feat}** and **{y_feat}**: **{r:.4f}**")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FEATURE INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬  Feature Insights":
    section("🔬 Feature Insights")

    col_sel, col_stats = st.columns([1, 1.6])

    with col_sel:
        feat = st.selectbox("Select a feature:", feature_cols)
        b_vals = df[df["diagnosis"] == "B"][feat]
        m_vals = df[df["diagnosis"] == "M"][feat]

        rows = [
            ("Mean",             round(df[feat].mean(), 4)),
            ("Median",           round(df[feat].median(), 4)),
            ("Std Dev",          round(df[feat].std(),  4)),
            ("Min",              round(df[feat].min(),  4)),
            ("Max",              round(df[feat].max(),  4)),
            ("Benign mean",      round(b_vals.mean(),   4)),
            ("Malignant mean",   round(m_vals.mean(),   4)),
            ("Δ (Malignant−Benign)", round(m_vals.mean() - b_vals.mean(), 4)),
            ("% difference",     f"{(m_vals.mean()-b_vals.mean())/b_vals.mean()*100:.1f}%"),
        ]
        stat_df = pd.DataFrame(rows, columns=["Statistic", "Value"])
        st.dataframe(stat_df, use_container_width=True, hide_index=True)

    with col_stats:
        fig, ax = make_fig(figsize=(6, 3.8))
        sns.violinplot(data=df, x="diagnosis_label", y=feat, ax=ax,
                       palette={"Benign": "#3B82F6", "Malignant": "#EF4444"},
                       inner="box")
        ax.set_title(f"Distribution: {feat}", fontsize=11, fontweight="bold")
        ax.set_xlabel(""); ax.set_ylabel(feat)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")

    # ── Random Forest Feature Importance ─────────────────────────────────────
    section("🌲 Top 15 Most Important Features — Random Forest")

    if rf_imp.empty:
        st.warning("⚠️ Feature importances not found. Run `python train_models.py` first.")
    else:
        top15 = rf_imp.head(15)
        fig, ax = make_fig(figsize=(9, 5.5))
        colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(top15)))[::-1]
        bars = ax.barh(range(len(top15)), top15.values[::-1],
                       color=colors[::-1], edgecolor="white")
        ax.set_yticks(range(len(top15)))
        ax.set_yticklabels([f.replace("_", " ").title() for f in top15.index[::-1]],
                           fontsize=9)
        ax.set_xlabel("Importance Score", fontsize=10)
        ax.set_title("Top 15 Feature Importances (Random Forest)",
                     fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.4)
        for bar, val in zip(bars, top15.values[::-1]):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("**Top 15 features table:**")
        imp_df = top15.reset_index()
        imp_df.columns = ["Feature", "Importance Score"]
        imp_df["Rank"] = range(1, len(imp_df) + 1)
        imp_df = imp_df[["Rank", "Feature", "Importance Score"]]
        st.dataframe(imp_df, use_container_width=True, hide_index=True)

        st.info(
            "ℹ️ Feature importance indicates patterns learned from this dataset. "
            "It does not imply medical causation."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Model Performance":
    section("🤖 Machine Learning — Model Performance")

    if not models_trained():
        st.error("⚠️ Models not found. Run `python train_models.py` first, then restart the app.")
        st.stop()

    # ── Comparison table ──────────────────────────────────────────────────────
    section("📊 Performance Comparison Table")
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    comp_rows = []
    for key, label in MODEL_LABELS.items():
        if key in metrics:
            m = metrics[key]
            comp_rows.append({
                "Model":     label,
                "Accuracy":  f"{m['accuracy']*100:.2f}%",
                "Precision": f"{m['precision']*100:.2f}%",
                "Recall":    f"{m['recall']*100:.2f}%",
                "F1-Score":  f"{m['f1']*100:.2f}%",
                "ROC-AUC":   f"{m['roc_auc']:.4f}",
            })
    if comp_rows:
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # ── Model cards ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section("📋 Model Cards")
    card_cols = st.columns(3)
    for i, (key, label) in enumerate(MODEL_LABELS.items()):
        if key not in metrics:
            continue
        m = metrics[key]
        with card_cols[i]:
            st.markdown(f"""
            <div class="model-card">
                <h4>{label}</h4>
                <span class="metric-pill">Acc {m['accuracy']*100:.1f}%</span>
                <span class="metric-pill">Prec {m['precision']*100:.1f}%</span>
                <span class="metric-pill">Rec {m['recall']*100:.1f}%</span>
                <span class="metric-pill">F1 {m['f1']*100:.1f}%</span>
                <span class="metric-pill">AUC {m['roc_auc']:.3f}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Confusion matrices ────────────────────────────────────────────────────
    section("🧩 Confusion Matrices")
    cm_cols = st.columns(3)
    for i, (key, label) in enumerate(MODEL_LABELS.items()):
        if key not in metrics:
            continue
        cm = np.array(metrics[key]["confusion_matrix"])
        fig, ax = make_fig(figsize=(4.5, 3.8))
        plot_confusion_matrix(cm, ax, title=label)
        plt.tight_layout()
        with cm_cols[i]:
            st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("""
    **Reading the confusion matrix:**  
    - **TN** = Benign correctly predicted as Benign  
    - **TP** = Malignant correctly predicted as Malignant  
    - **FP** = Benign incorrectly predicted as Malignant (false alarm)  
    - **FN** = Malignant incorrectly predicted as Benign (**most critical error**)
    """)

    # ── ROC curves ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section("📈 ROC Curves")
    fig, ax = make_fig(figsize=(7, 5))
    plot_roc_curves(metrics, ax)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Metric explanations ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📖 What do these metrics mean?"):
        st.markdown("""
| Metric | What it measures |
|--------|-----------------|
| **Accuracy** | Overall % of correct predictions |
| **Precision** | Of all cases predicted Malignant, what % are truly Malignant? |
| **Recall** | Of all actual Malignant cases, what % did the model correctly identify? |
| **F1-Score** | Harmonic mean of Precision and Recall — balanced view |
| **ROC-AUC** | Area under the ROC curve; 1.0 = perfect, 0.5 = random |

> **Recall is especially important** in this type of problem. A low recall means the model is 
> missing actual malignant cases (false negatives). In a real clinical setting, this would be 
> the most dangerous type of error — though this model is **not** a clinical tool.
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Prediction":
    section("🔮 Interactive Prediction")
    disclaimer()
    st.markdown("<br>", unsafe_allow_html=True)

    if not models_trained():
        st.error("⚠️ Models not found. Run `python train_models.py` first, then restart the app.")
        st.stop()

    # ── Model selector ────────────────────────────────────────────────────────
    model_choice = st.selectbox(
        "Select a model for prediction:",
        ["Logistic Regression", "Decision Tree", "Random Forest"],
    )
    MODEL_KEY_MAP = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree":       "decision_tree",
        "Random Forest":       "random_forest",
    }
    model_key = MODEL_KEY_MAP[model_choice]

    # ── Feature groups ────────────────────────────────────────────────────────
    mean_feats  = [f for f in feature_cols if f.endswith("_mean")]
    se_feats    = [f for f in feature_cols if f.endswith("_se")]
    worst_feats = [f for f in feature_cols if f.endswith("_worst")]

    # Use dataset medians as defaults (gives realistic inputs)
    defaults = df[feature_cols].median().to_dict()

    inputs = {}

    st.markdown("### Enter tumor measurements")
    st.markdown("*Default values are the dataset medians. Adjust as needed.*")

    with st.expander("📐 Mean Measurements (radius, texture, perimeter…)", expanded=True):
        cols = st.columns(2)
        for i, feat in enumerate(mean_feats):
            col = cols[i % 2]
            inputs[feat] = col.number_input(
                feat.replace("_mean", "").replace("_", " ").title() + " (mean)",
                value=float(round(defaults[feat], 4)),
                format="%.4f",
                key=f"inp_{feat}",
            )

    with st.expander("📏 Standard Error Measurements", expanded=False):
        cols = st.columns(2)
        for i, feat in enumerate(se_feats):
            col = cols[i % 2]
            inputs[feat] = col.number_input(
                feat.replace("_se", "").replace("_", " ").title() + " (SE)",
                value=float(round(defaults[feat], 4)),
                format="%.4f",
                key=f"inp_{feat}",
            )

    with st.expander("⚠️ Worst / Largest Measurements", expanded=False):
        cols = st.columns(2)
        for i, feat in enumerate(worst_feats):
            col = cols[i % 2]
            inputs[feat] = col.number_input(
                feat.replace("_worst", "").replace("_", " ").title() + " (worst)",
                value=float(round(defaults[feat], 4)),
                format="%.4f",
                key=f"inp_{feat}",
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Predict button ────────────────────────────────────────────────────────
    if st.button("🔍  ANALYZE PREDICTION", use_container_width=True, type="primary"):
        # Build input array in the exact same feature order as training
        input_array = np.array([[inputs[f] for f in feature_cols]])

        pipeline = load_pipeline(model_key)
        if pipeline is None:
            st.error("Model file not found. Run `python train_models.py` first.")
        else:
            pred = pipeline.predict(input_array)[0]
            prob = pipeline.predict_proba(input_array)[0]

            st.markdown("<br>", unsafe_allow_html=True)
            res_col, prob_col = st.columns([1, 1.2])

            with res_col:
                if pred == 1:
                    st.markdown(
                        '<div class="pred-malignant">🔴 Malignant'
                        '<div class="pred-label">The model predicts a malignant tumor</div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="pred-benign">🟢 Benign'
                        '<div class="pred-label">The model predicts a benign tumor</div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )

            with prob_col:
                st.markdown("**Prediction probabilities:**")
                fig, ax = make_fig(figsize=(5, 2.8))
                classes = ["Benign", "Malignant"]
                colors  = ["#3B82F6", "#EF4444"]
                bars = ax.barh(classes, [prob[0], prob[1]],
                               color=colors, edgecolor="white", height=0.4)
                ax.set_xlim([0, 1])
                ax.set_xlabel("Probability", fontsize=9)
                ax.set_title(f"Probabilities — {model_choice}", fontsize=10, fontweight="bold")
                for bar, p in zip(bars, [prob[0], prob[1]]):
                    ax.text(p + 0.01, bar.get_y() + bar.get_height() / 2,
                            f"{p*100:.1f}%", va="center", fontsize=10, fontweight="bold")
                ax.grid(axis="x", alpha=0.35)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**Model used:** {model_choice}")
            m = metrics.get(model_key, {})
            if m:
                st.markdown(
                    f"Test-set performance: "
                    f'<span class="metric-pill">Acc {m["accuracy"]*100:.1f}%</span>'
                    f'<span class="metric-pill">Rec {m["recall"]*100:.1f}%</span>'
                    f'<span class="metric-pill">AUC {m["roc_auc"]:.3f}</span>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            disclaimer()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — ABOUT PROJECT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️  About Project":
    section("ℹ️ About This Project")

    c1, c2 = st.columns([1.6, 1])
    with c1:
        st.markdown("""
### Project Title
**Breast Cancer Diagnosis: Exploratory Data Analysis and Predictive Modeling Using the Wisconsin Diagnostic Dataset**

### Objective
This project applies data analytics and machine-learning techniques to the Breast Cancer Wisconsin (Diagnostic) Dataset to:
- Identify measurable tumor characteristics that differ between benign and malignant cases
- Build and compare classification models
- Visualize findings in an interactive dashboard
- Demonstrate a complete data analytics pipeline

### Dataset
| Field | Value |
|-------|-------|
| Name | Breast Cancer Wisconsin (Diagnostic) Data Set |
| Source | [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) |
| Records | 569 patient samples |
| Features | 30 numeric diagnostic measurements |
| Target | diagnosis — M (Malignant) / B (Benign) |
        """)

    with c2:
        st.markdown("### Project Deliverables")
        deliverables = [
            ("📓", "Jupyter Notebook",        "EDA + ML pipeline"),
            ("🌐", "Streamlit App (this)",    "Interactive dashboard"),
            ("📄", "Project Report (.docx)",  "Full academic report"),
            ("📋", "README.md",               "Setup & run guide"),
            ("📦", "requirements.txt",         "Python dependencies"),
            ("🤖", "Trained Models (.pkl)",    "Joblib-serialized pipelines"),
        ]
        for icon, name, desc in deliverables:
            st.markdown(
                f'<div class="model-card" style="margin-bottom:.5rem;">'
                f'{icon} <strong>{name}</strong><br>'
                f'<span style="font-size:.82rem;color:#64748B;">{desc}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    if models_trained():
        section("📊 Actual Results (from trained models on test set)")
        comp_rows = []
        for key, label in MODEL_LABELS.items():
            if key in metrics:
                m = metrics[key]
                comp_rows.append({
                    "Model":     label,
                    "Accuracy":  f"{m['accuracy']*100:.2f}%",
                    "Precision": f"{m['precision']*100:.2f}%",
                    "Recall":    f"{m['recall']*100:.2f}%",
                    "F1-Score":  f"{m['f1']*100:.2f}%",
                    "ROC-AUC":   f"{m['roc_auc']:.4f}",
                })
        if comp_rows:
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
        if not rf_imp.empty:
            st.markdown("**Top 5 Most Important Features (Random Forest):**")
            for i, (feat, score) in enumerate(rf_imp.head(5).items(), 1):
                st.markdown(f"  {i}. `{feat}` — importance score: **{score:.4f}**")

    st.markdown("---")

    st.markdown("""
### Key Findings (from actual analysis)
1. **Dataset:** 569 samples, 30 features. 357 Benign (62.7%), 212 Malignant (37.3%).
2. **No missing values** in any feature column. `Unnamed: 32` was an all-null artifact.
3. **Most discriminating features:** `area_worst`, `concave_points_worst`, `radius_worst`, `perimeter_worst`, and `concave_points_mean` show the greatest separation between classes.
4. **Strong multicollinearity** among size-related features (radius, perimeter, area — r > 0.99), expected from their geometric relationship.
5. **All three models** performed well. Logistic Regression and Random Forest achieved ROC-AUC > 0.99.
6. **Recall focus:** The Random Forest achieved 0 false positives on the test set (Precision = 1.0), with 3 false negatives.
7. Cross-validation results were consistent with test-set performance.

### Limitations
- Small dataset (569 samples) from a single institution.
- Binary classification only; real diagnosis involves more factors.
- No external validation on an independent dataset.
- No clinical review. **Not a medical diagnostic tool.**

### Disclaimer
""")
    disclaimer()
    st.markdown("""
---
<div style='text-align:center;color:#94A3B8;font-size:.8rem;margin-top:1.5rem;'>
    Made with IBM Bob
</div>
""", unsafe_allow_html=True)
