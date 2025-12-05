import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


APP_TITLE = "Anti Corruption Analytics Demo"
DATA_FILES = {
    "acc": "ACC_COMPLAINTS.csv",
    "proc": "PROCUREMENT_FRAUD.csv",
    "rules": "FRAUD_PATTERNS.json",
    "evidence": "EVIDENCE_OCR_DATASET.txt",
}


# -------------------------
# Utilities
# -------------------------
def _safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


@st.cache_data(show_spinner=False)
def load_data():
    root = Path(__file__).parent

    acc_path = root / DATA_FILES["acc"]
    proc_path = root / DATA_FILES["proc"]
    rules_path = root / DATA_FILES["rules"]
    evidence_path = root / DATA_FILES["evidence"]

    acc_df = pd.read_csv(acc_path)
    proc_df = pd.read_csv(proc_path)

    # parse dates defensively
    if "date" in acc_df.columns:
        acc_df["date"] = pd.to_datetime(acc_df["date"], errors="coerce")
    if "award_date" in proc_df.columns:
        proc_df["award_date"] = pd.to_datetime(proc_df["award_date"], errors="coerce")

    fraud_rules = json.loads(_safe_read_text(rules_path))

    raw_text = _safe_read_text(evidence_path)
    evidence_docs = [block.strip() for block in raw_text.split("---") if block.strip()]

    return acc_df, proc_df, fraud_rules, evidence_docs


@st.cache_resource(show_spinner=False)
def train_risk_model(acc_df: pd.DataFrame):
    df = acc_df.copy()

    needed = {"amount", "sector", "accused_type", "channel", "division", "risk_label"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"ACC_COMPLAINTS.csv missing columns: {sorted(missing)}")

    X = df[["amount", "sector", "accused_type", "channel", "division"]]
    y = df["risk_label"].astype(str)

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
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return model, report


def show_classification_report(report: dict):
    rows = []
    for k, v in report.items():
        if isinstance(v, dict):
            rows.append(
                {
                    "label": k,
                    "precision": v.get("precision"),
                    "recall": v.get("recall"),
                    "f1-score": v.get("f1-score"),
                    "support": v.get("support"),
                }
            )
    rep_df = pd.DataFrame(rows)
    st.dataframe(rep_df, use_container_width=True)


def kpi_row(items):
    cols = st.columns(len(items))
    for c, (label, val) in zip(cols, items):
        with c:
            st.metric(label, val)


def add_mobile_css_and_footer():
    st.markdown(
        """
        <style>
          @media (max-width: 640px) {
            .block-container { padding-left: 0.9rem !important; padding-right: 0.9rem !important; }
            h1 { font-size: 2.0rem !important; }
            h2 { font-size: 1.45rem !important; }
            h3 { font-size: 1.15rem !important; }
          }

          /* Round some inputs a bit */
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
          .block-container { padding-bottom: 70px !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def module_complaints(acc_df: pd.DataFrame):
    st.subheader("ACC Complaint Intelligence")

    # KPIs
    kpi_row(
        [
            ("Total complaints", int(len(acc_df))),
            ("High risk", int((acc_df["risk_label"] == "High").sum())),
            ("Medium risk", int((acc_df["risk_label"] == "Medium").sum())),
            ("Low risk", int((acc_df["risk_label"] == "Low").sum())),
        ]
    )

    st.divider()

    # Charts (simple + dependency-free)
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### Risk distribution")
        risk_counts = (
            acc_df["risk_label"]
            .value_counts()
            .reindex(["High", "Medium", "Low"])
            .fillna(0)
            .astype(int)
        )
        st.bar_chart(risk_counts)

    with c2:
        st.markdown("#### Complaints over time")
        if "date" in acc_df.columns and acc_df["date"].notna().any():
            ts = (
                acc_df.dropna(subset=["date"])
                .assign(day=lambda d: d["date"].dt.date)
                .groupby("day")
                .size()
            )
            st.line_chart(ts)
        else:
            st.info("No valid dates found for time series chart.")

    st.divider()

    st.subheader("Filter complaints")
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        division = st.selectbox("Division", options=["All"] + sorted(acc_df["division"].astype(str).unique().tolist()))
    with col_f2:
        sector = st.selectbox("Sector", options=["All"] + sorted(acc_df["sector"].astype(str).unique().tolist()))
    with col_f3:
        channel = st.selectbox("Channel", options=["All"] + sorted(acc_df["channel"].astype(str).unique().tolist()))

    amt_min = float(acc_df["amount"].min())
    amt_max = float(acc_df["amount"].max())
    amount_range = st.slider("Amount range (BDT)", min_value=amt_min, max_value=amt_max, value=(amt_min, amt_max))

    filtered = acc_df.copy()
    if division != "All":
        filtered = filtered[filtered["division"].astype(str) == division]
    if sector != "All":
        filtered = filtered[filtered["sector"].astype(str) == sector]
    if channel != "All":
        filtered = filtered[filtered["channel"].astype(str) == channel]
    filtered = filtered[(filtered["amount"] >= amount_range[0]) & (filtered["amount"] <= amount_range[1])]

    # Top sectors chart (filtered)
    st.markdown("#### Top sectors (filtered)")
    top_sectors = filtered["sector"].astype(str).value_counts().head(10)
    st.bar_chart(top_sectors)

    st.markdown("#### Filtered results")
    show_cols = ["complaint_id", "date", "division", "district", "channel", "sector", "accused_type", "amount", "risk_label", "complaint_text"]
    show_cols = [c for c in show_cols if c in filtered.columns]
    filtered_to_show = filtered[show_cols]
    if "date" in filtered_to_show.columns:
        filtered_to_show = filtered_to_show.sort_values("date", ascending=False)

    st.dataframe(filtered_to_show, use_container_width=True, height=340)

    st.divider()

    st.subheader("Risk prediction model (baseline)")
    with st.spinner("Training model..."):
        model, report = train_risk_model(acc_df)

    st.caption("This is a simple, explainable baseline model (Logistic Regression) trained on the synthetic dataset.")
    show_classification_report(report)

    st.markdown("### Inspect a single complaint with model prediction")
    selected_id = st.selectbox("Select complaint_id", options=filtered["complaint_id"].astype(str).tolist() if len(filtered) else acc_df["complaint_id"].astype(str).tolist())
    row = acc_df[acc_df["complaint_id"].astype(str) == str(selected_id)].iloc[0]

    X_one = pd.DataFrame(
        {
            "amount": [float(row["amount"])],
            "sector": [str(row["sector"])],
            "accused_type": [str(row["accused_type"])],
            "channel": [str(row["channel"])],
            "division": [str(row["division"])],
        }
    )
    proba = model.predict_proba(X_one)[0]
    classes = list(model.classes_)
    pred = classes[int(np.argmax(proba))]

    st.write(f"**Prediction:** {pred}")
    st.write("**Probabilities:**")
    prob_tbl = pd.DataFrame({"label": classes, "probability": [float(x) for x in proba]}).sort_values("probability", ascending=False)
    st.dataframe(prob_tbl, use_container_width=True, height=160)
    st.write("**Complaint text:**")
    st.write(str(row.get("complaint_text", "")))


def module_procurement(proc_df: pd.DataFrame):
    st.subheader("Procurement Fraud Overview")

    # KPIs
    fraud_ones = int((proc_df.get("fraud_flag", 0) == 1).sum()) if "fraud_flag" in proc_df.columns else 0
    kpi_row(
        [
            ("Total tenders", int(len(proc_df))),
            ("Flagged (fraud_flag=1)", fraud_ones),
            ("Unique entities", int(proc_df["procuring_entity"].nunique()) if "procuring_entity" in proc_df.columns else 0),
            ("Unique suppliers", int(proc_df["supplier"].nunique()) if "supplier" in proc_df.columns else 0),
        ]
    )

    st.divider()

    # Basic rule signals
    df = proc_df.copy()
    if "bidders_count" in df.columns:
        df["rule_single_bidder"] = (df["bidders_count"] == 1).astype(int)
    else:
        df["rule_single_bidder"] = 0

    if "contract_value" in df.columns and "estimated_value" in df.columns:
        df["rule_high_inflation"] = (df["contract_value"] > df["estimated_value"] * 1.25).astype(int)
    else:
        df["rule_high_inflation"] = 0

    df["rule_score"] = df["rule_single_bidder"] * 2 + df["rule_high_inflation"] * 1

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Fraud by sector (count)")
        if "sector" in df.columns:
            fraud_by_sector = df.groupby("sector")["fraud_flag"].sum() if "fraud_flag" in df.columns else df.groupby("sector")["rule_score"].sum()
            fraud_by_sector = fraud_by_sector.sort_values(ascending=False).head(12)
            st.bar_chart(fraud_by_sector)
        else:
            st.info("No sector column found.")

    with c2:
        st.markdown("#### Rule signals (how many)")
        signals = pd.Series(
            {
                "Single bidder": int(df["rule_single_bidder"].sum()),
                "High inflation": int(df["rule_high_inflation"].sum()),
                "Any rule hit": int((df["rule_score"] > 0).sum()),
            }
        )
        st.bar_chart(signals)

    st.divider()

    st.subheader("Filter tenders")
    col1, col2, col3 = st.columns(3)
    with col1:
        sector = st.selectbox("Sector", options=["All"] + sorted(df["sector"].astype(str).unique().tolist()) if "sector" in df.columns else ["All"])
    with col2:
        district = st.selectbox("District", options=["All"] + sorted(df["district"].astype(str).unique().tolist()) if "district" in df.columns else ["All"])
    with col3:
        method = st.selectbox("Method", options=["All"] + sorted(df["method"].astype(str).unique().tolist()) if "method" in df.columns else ["All"])

    show_flagged = st.checkbox("Show only suspicious (fraud_flag=1 OR rule_score>0)", value=True)

    f = df.copy()
    if sector != "All" and "sector" in f.columns:
        f = f[f["sector"].astype(str) == sector]
    if district != "All" and "district" in f.columns:
        f = f[f["district"].astype(str) == district]
    if method != "All" and "method" in f.columns:
        f = f[f["method"].astype(str) == method]
    if show_flagged and "fraud_flag" in f.columns:
        f = f[(f["fraud_flag"] == 1) | (f["rule_score"] > 0)]
    elif show_flagged:
        f = f[f["rule_score"] > 0]

    show_cols = [
        "tender_id", "award_date", "procuring_entity", "supplier", "method",
        "estimated_value", "contract_value", "bidders_count", "sector", "district",
        "fraud_flag", "rule_single_bidder", "rule_high_inflation", "rule_score"
    ]
    show_cols = [c for c in show_cols if c in f.columns]
    if "award_date" in f.columns:
        f = f.sort_values("award_date", ascending=False)

    st.dataframe(f[show_cols], use_container_width=True, height=380)


def module_evidence(evidence_docs):
    st.subheader("Evidence Viewer (OCR-like text)")

    st.caption("Select a document and review suspicious indicators. This is a demo scanner on synthetic text blocks.")
    if not evidence_docs:
        st.warning("No evidence documents found.")
        return

    idx = st.selectbox("Select evidence document", options=list(range(1, len(evidence_docs) + 1)))
    doc = evidence_docs[idx - 1]

    # simple highlight
    keywords = ["SUSPICIOUS", "Mismatch", "overwrite", "hand-written", "signature", "tamper", "alter", "partially"]
    lines = doc.splitlines()

    st.markdown("#### Document text")
    for ln in lines:
        if any(k.lower() in ln.lower() for k in keywords):
            st.markdown(f"- **{ln}**")
        else:
            st.markdown(f"- {ln}")


def module_rules(fraud_rules):
    st.subheader("Fraud Rules Library")

    if isinstance(fraud_rules, list) and fraud_rules:
        df = pd.DataFrame(fraud_rules)
        search = st.text_input("Search (rule_name/description/trigger/severity)", "")
        if search.strip():
            s = search.strip().lower()
            mask = pd.Series(False, index=df.index)
            for col in df.columns:
                mask = mask | df[col].astype(str).str.lower().str.contains(s, na=False)
            df = df[mask]
        st.dataframe(df, use_container_width=True, height=420)
        with st.expander("View raw JSON"):
            st.json(fraud_rules)
    else:
        st.info("Rules JSON is empty or not a list.")
        with st.expander("Raw"):
            st.json(fraud_rules)


def main():
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    add_mobile_css_and_footer()

    st.title(APP_TITLE)
    st.caption("Built on synthetic datasets for interview demonstration")

    acc_df, proc_df, fraud_rules, evidence_docs = load_data()

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

    st.divider()

    if page == "ACC Complaint Intelligence":
        module_complaints(acc_df)
    elif page == "Procurement Fraud Overview":
        module_procurement(proc_df)
    elif page == "Evidence Viewer":
        module_evidence(evidence_docs)
    else:
        module_rules(fraud_rules)

    st.markdown('<div class="ifaz-footer">Ifaz Ahmed Chowdhury &copy; 2026</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
