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

# -------------------------
# Data loading
# -------------------------

@st.cache_data
def load_data():
    acc_df = pd.read_csv("ACC_COMPLAINTS.csv")
    proc_df = pd.read_csv("PROCUREMENT_FRAUD.csv")

    with open("FRAUD_PATTERNS.json", "r", encoding="utf-8") as f:
        fraud_rules = json.load(f)

    with open("EVIDENCE_OCR_DATASET.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Split OCR documents by separator ---
    docs = [block.strip() for block in raw_text.split("---") if block.strip()]

    return acc_df, proc_df, fraud_rules, docs


# -------------------------
# Model training for risk_label
# -------------------------

@st.cache_resource
def train_risk_model(acc_df: pd.DataFrame):
    df = acc_df.copy()

    # Features and target
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


# Utility to show classification report nicely
def show_classification_report(report_dict):
    st.subheader("Risk label classification quality")
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
        rep_df = pd.DataFrame(rows)
        st.table(rep_df)


# -------------------------
# Rules engine for procurement fraud (optional)
# -------------------------

def apply_simple_rules(proc_df: pd.DataFrame) -> pd.DataFrame:
    df = proc_df.copy()

    # Basic rule signals
    df["rule_single_bidder"] = (df["bidders_count"] == 1).astype(int)
    df["rule_high_inflation"] = (
        df["contract_value"] > df["estimated_value"] * 1.25
    ).astype(int)

    # Rough flag score
    df["rule_score"] = df["rule_single_bidder"] * 2 + df["rule_high_inflation"] * 1

    return df


# -------------------------
# Streamlit UI
# -------------------------

def main():
    st.set_page_config(
        page_title="Anti Corruption Analytics Demo",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    
    # --- Mobile-first CSS tweaks ---
    st.markdown(
        """
        <style>
          /* Reduce side padding a bit on small screens */
          @media (max-width: 640px) {
            .block-container { padding-left: 0.9rem !important; padding-right: 0.9rem !important; }
            h1 { font-size: 2.0rem !important; }
            h2 { font-size: 1.4rem !important; }
            h3 { font-size: 1.1rem !important; }
          }

          /* Make widgets breathe a little */
          div[data-baseweb="select"] > div { border-radius: 12px; }
          .stTextInput input, .stNumberInput input, .stSelectbox div, .stTextArea textarea {
            border-radius: 12px !important;
          }

          /* Sticky footer */
          .ifaz-footer {
            position: fixed;
            left: 0; right: 0; bottom: 0;
            padding: 10px 12px;
            font-size: 13px;
            text-align: center;
            background: rgba(255,255,255,0.92);
            backdrop-filter: blur(8px);
            border-top: 1px solid rgba(0,0,0,0.08);
            z-index: 9999;
          }

          /* Keep content above footer */
          .block-container { padding-bottom: 70px !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
st.title("Anti Corruption Analytics Demo")
    st.caption("Built on synthetic datasets for interview demonstration")

    acc_df, proc_df, fraud_rules, evidence_docs = load_data()

    # Module navigation (mobile friendly)
    st.markdown("### Select module")
    page = st.selectbox(
        "Select module",
        [
            "ACC Complaint Intelligence",
            "Procurement Fraud Overview",
            "Evidence Viewer",
            "Fraud Rules Library",
        ],
    )
if page == "ACC Complaint Intelligence":
        show_acc_complaint_module(acc_df)
    elif page == "Procurement Fraud Overview":
        show_procurement_module(proc_df)
    elif page == "Evidence Viewer":
        show_evidence_module(evidence_docs)
    elif page == "Fraud Rules Library":
        show_fraud_rules_module(fraud_rules)


def show_acc_complaint_module(acc_df: pd.DataFrame):
    st.header("ACC Complaint Intelligence")

    st.markdown(
        """
This page demonstrates how incoming complaints can be
analyzed, filtered, and scored using data features and a simple ML model.
"""
    )

    # Basic stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total complaints", len(acc_df))

    with col2:
        st.metric("High risk", int((acc_df["risk_label"] == "High").sum()))

    with col3:
        st.metric("Medium risk", int((acc_df["risk_label"] == "Medium").sum()))

    with col4:
        st.metric("Low risk", int((acc_df["risk_label"] == "Low").sum()))

    # Filters
    st.subheader("Filter complaints")

    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        division = st.selectbox(
            "Division",
            options=["All"] + sorted(acc_df["division"].unique().tolist()),
        )

    with col_f2:
        sector = st.selectbox(
            "Sector",
            options=["All"] + sorted(acc_df["sector"].unique().tolist()),
        )

    with col_f3:
        risk = st.selectbox(
            "Risk label",
            options=["All"] + sorted(acc_df["risk_label"].unique().tolist()),
        )

    filtered = acc_df.copy()

    if division != "All":
        filtered = filtered[filtered["division"] == division]
    if sector != "All":
        filtered = filtered[filtered["sector"] == sector]
    if risk != "All":
        filtered = filtered[filtered["risk_label"] == risk]

    st.write(f"Showing {len(filtered)} complaints after filter")

    st.dataframe(
        filtered[
            [
                "complaint_id",
                "date",
                "division",
                "district",
                "sector",
                "accused_type",
                "amount",
                "risk_label",
            ]
        ].sort_values("date", ascending=False),
        use_container_width=True,
        height=300,
    )

    # ML model
    st.subheader("Risk prediction model on complaint features")
    with st.spinner("Training model..."):
        model, report = train_risk_model(acc_df)

    show_classification_report(report)

    st.markdown("### Inspect single complaint with model prediction")

    selected_id = st.selectbox(
        "Select complaint id",
        options=filtered["complaint_id"].unique().tolist(),
    )

    row = acc_df[acc_df["complaint_id"] == selected_id].iloc[0]

    st.write("**Complaint text:**")
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

    col_a, col_b = st.columns(2)

    with col_a:
        st.write(f"True risk label: **{row['risk_label']}**")
        st.write(f"Model predicted: **{pred}**")

    with col_b:
        st.write("Prediction probabilities:")
        st.json({cls: round(p, 3) for cls, p in proba_dict.items()})


def show_procurement_module(proc_df: pd.DataFrame):
    st.header("Procurement Fraud Overview")

    st.markdown(
        """
This page summarizes tender data and highlights entries that look suspicious
according to simple rules and existing fraud flags.
"""
    )

    enhanced = apply_simple_rules(proc_df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total tenders", len(enhanced))

    with col2:
        st.metric("Flagged by dataset", int(enhanced["fraud_flag"].sum()))

    with col3:
        st.metric(
            "Single bidder cases",
            int(enhanced["rule_single_bidder"].sum()),
        )

    with col4:
        st.metric(
            "High inflation cases",
            int(enhanced["rule_high_inflation"].sum()),
        )

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        entity = st.selectbox(
            "Procuring entity",
            ["All"] + sorted(enhanced["procuring_entity"].unique().tolist()),
        )

    with col_f2:
        sector = st.selectbox(
            "Sector",
            ["All"] + sorted(enhanced["sector"].unique().tolist()),
        )

    with col_f3:
        only_suspicious = st.checkbox("Show suspicious only", value=False)

    df = enhanced.copy()

    if entity != "All":
        df = df[df["procuring_entity"] == entity]
    if sector != "All":
        df = df[df["sector"] == sector]
    if only_suspicious:
        df = df[(df["fraud_flag"] == 1) | (df["rule_score"] > 0)]

    st.write(f"Showing {len(df)} tenders after filter")

    st.dataframe(
        df[
            [
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
        ].sort_values("award_date", ascending=False),
        use_container_width=True,
        height=350,
    )


def show_evidence_module(evidence_docs):
    st.header("Evidence OCR Document Viewer")

    st.markdown(
        """
This page shows sample OCR text of government documents such as
invoices, vouchers and tender related papers with suspicious indicators.
"""
    )

    idx = st.slider(
        "Select document index",
        min_value=1,
        max_value=len(evidence_docs),
        value=1,
    )

    doc = evidence_docs[idx - 1]

    st.subheader(f"Document {idx}")
    st.text(doc)

    # Simple highlight for suspicious lines
    st.markdown("### Detected suspicious lines (simple text match)")

    suspicious_lines = [
        line for line in doc.splitlines() if "SUSPICIOUS" in line.upper() or "indicator" in line.lower()
    ]

    if suspicious_lines:
        for line in suspicious_lines:
            st.write(f"- {line}")
    else:
        st.write("No explicit suspicious line markers found in this sample.")


def show_fraud_rules_module(fraud_rules):
    st.header("Fraud Rules Library")

    st.markdown(
        """
The rules below represent a knowledge base of fraud indicators
that can be combined with ML models to produce risk scores.
"""
    )

    st.json(fraud_rules)


    # Footer
    st.markdown('<div class="ifaz-footer">Ifaz Ahmed Chowdhury &copy; 2026</div>', unsafe_allow_html=True)

if __name__ == "__main__":

    main()
