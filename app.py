import json
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


APP_TITLE = "Anti Corruption Analytics Demo"
APP_SUBTITLE = "Synthetic datasets plus a simple ML and rules engine for interview demonstration"
FOOTER_TEXT = "Ifaz Ahmed Chowdhury © 2026"


def inject_css():
    st.markdown(
        """
<style>
/* Hide Streamlit default UI bits */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Global spacing so fixed footer does not overlap content */
.block-container {padding-bottom: 3.5rem;}

/* Nicer sidebar */
section[data-testid="stSidebar"] {
  border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] .stRadio label {
  padding: 0.35rem 0.4rem;
  border-radius: 0.65rem;
}

/* Card-ish look for metrics */
div[data-testid="stMetric"] {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  padding: 0.85rem 0.9rem;
  border-radius: 1rem;
}
div[data-testid="stMetric"] > div {gap: 0.15rem;}

/* Tables and dataframes */
div[data-testid="stDataFrame"] {
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 1rem;
  overflow: hidden;
}

/* Subtle section header */
.k-section {
  padding: 0.85rem 1rem;
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.03);
  border-radius: 1rem;
  margin: 0.25rem 0 1rem 0;
}

/* Fixed footer */
.k-footer {
  position: fixed;
  left: 0;
  bottom: 0;
  width: 100%;
  background: rgba(0,0,0,0.35);
  backdrop-filter: blur(8px);
  border-top: 1px solid rgba(255,255,255,0.10);
  padding: 0.6rem 1rem;
  text-align: center;
  font-size: 0.9rem;
  z-index: 9999;
}
</style>
""",
        unsafe_allow_html=True,
    )


# -------------------------
# Data loading
# -------------------------

@st.cache_data
def load_data():
    try:
        acc_df = pd.read_csv("ACC_COMPLAINTS.csv")
        proc_df = pd.read_csv("PROCUREMENT_FRAUD.csv")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Missing dataset files. Keep ACC_COMPLAINTS.csv and PROCUREMENT_FRAUD.csv "
            "in the same folder as app.py when running or deploying."
        ) from e

    with open("FRAUD_PATTERNS.json", "r", encoding="utf-8") as f:
        fraud_rules = json.load(f)

    with open("EVIDENCE_OCR_DATASET.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    docs = [block.strip() for block in raw_text.split("---") if block.strip()]

    return acc_df, proc_df, fraud_rules, docs


# -------------------------
# Model training for risk_label
# -------------------------

@st.cache_resource
def train_risk_model(acc_df: pd.DataFrame):
    df = acc_df.copy()

    required_cols = ["amount", "sector", "accused_type", "channel", "division", "risk_label"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"ACC_COMPLAINTS.csv missing required column: {c}")

    X = df[["amount", "sector", "accused_type", "channel", "division"]]
    y = df["risk_label"]

    cat_cols = ["sector", "accused_type", "channel", "division"]
    num_cols = ["amount"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, multi_class="auto")),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return model, report


def show_classification_report(report_dict):
    st.subheader("Model quality (risk label)")

    rows = []
    for label, metrics in report_dict.items():
        if label in ["accuracy", "macro avg", "weighted avg"]:
            continue
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "label": label,
                "precision": round(metrics.get("precision", 0), 2),
                "recall": round(metrics.get("recall", 0), 2),
                "f1": round(metrics.get("f1-score", 0), 2),
                "support": int(metrics.get("support", 0)),
            }
        )
    if rows:
        rep_df = pd.DataFrame(rows).sort_values("label")
        st.dataframe(rep_df, use_container_width=True, hide_index=True)


# -------------------------
# Rules engine for procurement fraud (simple)
# -------------------------

def apply_simple_rules(proc_df: pd.DataFrame) -> pd.DataFrame:
    df = proc_df.copy()

    needed = ["bidders_count", "contract_value", "estimated_value"]
    for c in needed:
        if c not in df.columns:
            return df

    df["rule_single_bidder"] = (df["bidders_count"] == 1).astype(int)
    df["rule_high_inflation"] = (df["contract_value"] > df["estimated_value"] * 1.25).astype(int)

    df["rule_score"] = df["rule_single_bidder"] * 2 + df["rule_high_inflation"] * 1

    return df


# -------------------------
# UI modules
# -------------------------

def page_header():
    st.markdown(
        f"""
<div class="k-section">
  <div style="font-size: 2rem; font-weight: 800; line-height: 1.1;">{APP_TITLE}</div>
  <div style="opacity: 0.80; margin-top: 0.35rem;">{APP_SUBTITLE}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def show_acc_complaint_module(acc_df: pd.DataFrame):
    st.header("ACC Complaint Intelligence")

    st.markdown(
        "Explore complaint inflow, filter by region and sector, then test a simple model that predicts risk labels."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total complaints", len(acc_df))
    with col2:
        st.metric("High risk", int((acc_df["risk_label"] == "High").sum()))
    with col3:
        st.metric("Medium risk", int((acc_df["risk_label"] == "Medium").sum()))
    with col4:
        st.metric("Low risk", int((acc_df["risk_label"] == "Low").sum()))

    st.subheader("Quick charts")
    c1, c2 = st.columns(2)

    with c1:
        st.caption("Risk labels distribution")
        st.bar_chart(acc_df["risk_label"].value_counts(), use_container_width=True)

    with c2:
        st.caption("Top sectors by complaint count")
        st.bar_chart(acc_df["sector"].value_counts().head(10), use_container_width=True)

    st.divider()
    st.subheader("Filter complaints")

    col_f1, col_f2, col_f3, col_f4 = st.columns([1, 1, 1, 1])
    with col_f1:
        division = st.selectbox("Division", options=["All"] + sorted(acc_df["division"].unique().tolist()))
    with col_f2:
        district = st.selectbox("District", options=["All"] + sorted(acc_df["district"].unique().tolist()))
    with col_f3:
        sector = st.selectbox("Sector", options=["All"] + sorted(acc_df["sector"].unique().tolist()))
    with col_f4:
        risk = st.selectbox("Risk label", options=["All"] + sorted(acc_df["risk_label"].unique().tolist()))

    filtered = acc_df.copy()
    if division != "All":
        filtered = filtered[filtered["division"] == division]
    if district != "All":
        filtered = filtered[filtered["district"] == district]
    if sector != "All":
        filtered = filtered[filtered["sector"] == sector]
    if risk != "All":
        filtered = filtered[filtered["risk_label"] == risk]

    st.write(f"Showing {len(filtered)} complaints after filtering")

    st.dataframe(
        filtered[
            ["complaint_id", "date", "division", "district", "sector", "accused_type", "amount", "risk_label"]
        ].sort_values("date", ascending=False),
        use_container_width=True,
        height=340,
        hide_index=True,
    )

    st.divider()
    st.subheader("Risk prediction model")
    with st.spinner("Training model..."):
        model, report = train_risk_model(acc_df)

    show_classification_report(report)

    st.markdown("### Inspect one complaint with model prediction")

    ids = filtered["complaint_id"].unique().tolist()
    if not ids:
        st.info("No complaints in the current filter. Change filters to inspect a complaint.")
        return

    selected_id = st.selectbox("Complaint id", options=ids)
    row = acc_df[acc_df["complaint_id"] == selected_id].iloc[0]

    left, right = st.columns([1.4, 1])
    with left:
        st.write("**Complaint text**")
        st.write(row["complaint_text"])

    features = pd.DataFrame(
        [
            {
                "amount": row["amount"],
                "sector": row["sector"],
                "accused_type": row["accused_type"],
                "channel": row["channel"],
                "division": row["division"],
            }
        ]
    )

    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    proba_dict = dict(zip(model.classes_, proba))

    with right:
        st.write(f"True label: **{row['risk_label']}**")
        st.write(f"Predicted: **{pred}**")
        st.caption("Class probabilities")
        st.json({cls: float(np.round(p, 3)) for cls, p in proba_dict.items()})


def show_procurement_module(proc_df: pd.DataFrame):
    st.header("Procurement Fraud Overview")

    st.markdown(
        "Summarize tenders and highlight suspicious entries using simple rule signals and the dataset fraud flag."
    )

    enhanced = apply_simple_rules(proc_df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total tenders", len(enhanced))
    with col2:
        st.metric("Flagged by dataset", int(enhanced.get("fraud_flag", pd.Series([0]*len(enhanced))).sum()))
    with col3:
        st.metric("Single bidder cases", int(enhanced.get("rule_single_bidder", pd.Series([0]*len(enhanced))).sum()))
    with col4:
        st.metric("High inflation cases", int(enhanced.get("rule_high_inflation", pd.Series([0]*len(enhanced))).sum()))

    st.subheader("Quick charts")
    c1, c2 = st.columns(2)
    with c1:
        if "method" in enhanced.columns:
            st.caption("Procurement methods")
            st.bar_chart(enhanced["method"].value_counts(), use_container_width=True)
        else:
            st.info("No method column found in dataset.")
    with c2:
        if "fraud_flag" in enhanced.columns:
            st.caption("Fraud flag distribution")
            st.bar_chart(enhanced["fraud_flag"].value_counts(), use_container_width=True)
        else:
            st.info("No fraud_flag column found in dataset.")

    st.divider()
    col_f1, col_f2, col_f3, col_f4 = st.columns([1, 1, 1, 1])
    with col_f1:
        entity = st.selectbox("Procuring entity", ["All"] + sorted(enhanced["procuring_entity"].unique().tolist()))
    with col_f2:
        sector = st.selectbox("Sector", ["All"] + sorted(enhanced["sector"].unique().tolist()))
    with col_f3:
        district = st.selectbox("District", ["All"] + sorted(enhanced["district"].unique().tolist()))
    with col_f4:
        only_suspicious = st.checkbox("Show suspicious only", value=False)

    df = enhanced.copy()
    if entity != "All":
        df = df[df["procuring_entity"] == entity]
    if sector != "All":
        df = df[df["sector"] == sector]
    if district != "All":
        df = df[df["district"] == district]
    if only_suspicious:
        if "fraud_flag" in df.columns and "rule_score" in df.columns:
            df = df[(df["fraud_flag"] == 1) | (df["rule_score"] > 0)]
        elif "fraud_flag" in df.columns:
            df = df[df["fraud_flag"] == 1]

    st.write(f"Showing {len(df)} tenders after filtering")

    cols = [
        "tender_id",
        "procuring_entity",
        "supplier",
        "method",
        "estimated_value",
        "contract_value",
        "bidders_count",
        "district",
        "sector",
        "award_date",
        "fraud_flag",
        "rule_single_bidder",
        "rule_high_inflation",
        "rule_score",
    ]
    existing_cols = [c for c in cols if c in df.columns]

    if "award_date" in df.columns:
        df_sorted = df.sort_values("award_date", ascending=False)
    else:
        df_sorted = df

    st.dataframe(df_sorted[existing_cols], use_container_width=True, height=380, hide_index=True)


def show_evidence_module(evidence_docs):
    st.header("Evidence OCR Document Viewer")

    st.markdown("Browse sample OCR-like document text and surface suspicious lines using a simple keyword scan.")

    if not evidence_docs:
        st.info("No evidence docs found in EVIDENCE_OCR_DATASET.txt.")
        return

    idx = st.slider("Document index", min_value=1, max_value=len(evidence_docs), value=1)
    doc = evidence_docs[idx - 1]

    st.subheader(f"Document {idx}")
    st.code(doc, language="text")

    st.markdown("### Suspicious lines (simple match)")
    suspicious_lines = [
        line
        for line in doc.splitlines()
        if ("SUSPICIOUS" in line.upper())
        or ("indicator" in line.lower())
        or ("overwrite" in line.lower())
        or ("scanned" in line.lower())
        or ("does not match" in line.lower())
    ]
    if suspicious_lines:
        for line in suspicious_lines:
            st.write(f"- {line.strip()}")
    else:
        st.write("No suspicious keyword matches found in this document.")


def show_fraud_rules_module(fraud_rules):
    st.header("Fraud Rules Library")

    st.markdown("A small knowledge base of fraud indicators that can complement ML scoring.")

    if isinstance(fraud_rules, list) and fraud_rules:
        df = pd.DataFrame(fraud_rules)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.json(fraud_rules)


def render_footer():
    st.markdown(
        f"""<div class="k-footer">{FOOTER_TEXT}</div>""",
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_css()
    page_header()

    acc_df, proc_df, fraud_rules, evidence_docs = load_data()

    with st.sidebar:
        st.markdown("### Modules")
        page = st.radio(
            "Navigate",
            ["ACC Complaint Intelligence", "Procurement Fraud Overview", "Evidence Viewer", "Fraud Rules Library"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("Tip: Use filters to narrow down results fast.")

    if page == "ACC Complaint Intelligence":
        show_acc_complaint_module(acc_df)
    elif page == "Procurement Fraud Overview":
        show_procurement_module(proc_df)
    elif page == "Evidence Viewer":
        show_evidence_module(evidence_docs)
    elif page == "Fraud Rules Library":
        show_fraud_rules_module(fraud_rules)

    render_footer()


if __name__ == "__main__":
    main()
