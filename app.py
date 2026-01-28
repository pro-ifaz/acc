import json
import re
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


APP_TITLE = "ACC Anti Corruption Analytics Demo"


def enrich_complaints(acc_df: pd.DataFrame) -> pd.DataFrame:
    df = acc_df.copy()

    if "complaint_id" not in df.columns:
        df["complaint_id"] = [f"C-{i+1:04d}" for i in range(len(df))]

    df["amount"] = pd.to_numeric(df.get("amount", np.nan), errors="coerce")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "complaint_text" not in df.columns:
        df["complaint_text"] = ""
    text_series = df["complaint_text"].fillna("").astype(str)

    if "amount_log" not in df.columns:
        df["amount_log"] = np.log1p(df["amount"].fillna(0).clip(lower=0))

    if "text_length" not in df.columns:
        df["text_length"] = text_series.str.len()

    if "word_count" not in df.columns:
        df["word_count"] = text_series.apply(lambda s: len(re.findall(r"\w+", s)))

    if "amount_band" not in df.columns:
        amt = df["amount"].fillna(0).clip(lower=0)
        try:
            df["amount_band"] = pd.qcut(
                amt,
                q=4,
                labels=["Low", "Medium", "High", "Very High"],
                duplicates="drop",
            ).astype(str)
        except Exception:
            bins = [-1, 5000, 20000, 100000, float("inf")]
            labels = ["Low", "Medium", "High", "Very High"]
            df["amount_band"] = pd.cut(amt, bins=bins, labels=labels).astype(str)

    for col in ["sector", "accused_type", "channel", "division", "district", "risk_label"]:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].fillna("Unknown").astype(str)

    return df


def enrich_procurement(proc_df: pd.DataFrame) -> pd.DataFrame:
    df = proc_df.copy()

    for col in ["estimated_value", "contract_value", "bidders_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "estimated_value" not in df.columns:
        df["estimated_value"] = np.nan
    if "contract_value" not in df.columns:
        df["contract_value"] = np.nan
    if "bidders_count" not in df.columns:
        df["bidders_count"] = np.nan

    if "award_date" in df.columns:
        df["award_date"] = pd.to_datetime(df["award_date"], errors="coerce")

    if "inflation_ratio" not in df.columns:
        df["inflation_ratio"] = df["contract_value"] / df["estimated_value"].replace({0: np.nan})

    if "is_single_bidder" not in df.columns:
        df["is_single_bidder"] = (df["bidders_count"] == 1).fillna(False).astype(int)

    if "is_high_inflation" not in df.columns:
        df["is_high_inflation"] = (df["inflation_ratio"] > 1.25).fillna(False).astype(int)

    if "rule_risk_score" not in df.columns:
        df["rule_risk_score"] = df["is_single_bidder"] * 2 + df["is_high_inflation"] * 1

    for col in ["tender_id", "procuring_entity", "supplier", "method", "district", "sector"]:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].fillna("Unknown").astype(str)

    if "fraud_flag" in df.columns:
        df["fraud_flag"] = pd.to_numeric(df["fraud_flag"], errors="coerce").fillna(0).astype(int)
    else:
        df["fraud_flag"] = 0

    return df


def enrich_evidence(ev_df: pd.DataFrame) -> pd.DataFrame:
    df = ev_df.copy()
    if "sample_id" not in df.columns:
        df["sample_id"] = [f"E-{i+1:04d}" for i in range(len(df))]

    for c in ["flag_overwritten", "flag_mismatch_total", "flag_duplicate_invoice", "flag_impossible_date"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        else:
            df[c] = 0

    for col in ["doc_type", "issuing_entity", "date_raw", "red_flag", "raw_text"]:
        if col not in df.columns:
            df[col] = ""

    return df


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


@st.cache_data(show_spinner=False)
def load_data():
    root = Path(__file__).parent

    def must_exist(p: Path, label: str):
        if not p.exists():
            raise FileNotFoundError(f"Missing {label}: {p.name}. Keep it beside app.py.")
        return p

    acc_df = pd.read_csv(must_exist(root / "ACC_COMPLAINTS.csv", "complaints dataset"))
    proc_df = pd.read_csv(must_exist(root / "PROCUREMENT_FRAUD.csv", "procurement dataset"))
    rules = json.loads(_read_text(must_exist(root / "FRAUD_PATTERNS.json", "fraud rules json")))

    ev_csv = root / "EVIDENCE_OCR_DATASET.csv"
    ev_txt = root / "EVIDENCE_OCR_DATASET.txt"
    if ev_csv.exists():
        ev_df = pd.read_csv(ev_csv)
    elif ev_txt.exists():
        raw = _read_text(ev_txt)
        blocks = [b.strip() for b in raw.split("---") if b.strip()]
        ev_df = pd.DataFrame({"sample_id": [f"E-{i+1:04d}" for i in range(len(blocks))], "raw_text": blocks})
    else:
        ev_df = pd.DataFrame(columns=["sample_id", "raw_text"])

    return enrich_complaints(acc_df), enrich_procurement(proc_df), enrich_evidence(ev_df), rules


@st.cache_resource(show_spinner=False)
def train_risk_model(acc_df: pd.DataFrame):
    df = enrich_complaints(acc_df)

    feature_cols_cat = ["sector", "accused_type", "channel", "division", "amount_band"]
    feature_cols_num = ["amount", "amount_log", "text_length", "word_count"]

    X = df[feature_cols_cat + feature_cols_num]
    y = df["risk_label"].astype(str)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat),
            ("num", "passthrough", feature_cols_num),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", LogisticRegression(max_iter=2000, multi_class="auto")),
        ]
    )

    strat = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return model, report


def df_value_counts(series: pd.Series, limit=12) -> pd.Series:
    vc = series.dropna().astype(str).value_counts()
    if limit and len(vc) > limit:
        top = vc.head(limit).copy()
        top.loc["Other"] = int(vc.iloc[limit:].sum())
        return top
    return vc


def inject_css():
    st.markdown(
        """
        <style>
          @media (max-width: 640px) {
            .block-container { padding-left: 0.9rem !important; padding-right: 0.9rem !important; }
            h1 { font-size: 2.0rem !important; }
            h2 { font-size: 1.45rem !important; }
            h3 { font-size: 1.15rem !important; }
          }
          .ifaz-footer{
            position:fixed; left:0; right:0; bottom:0;
            padding:10px 12px; font-size:13px; text-align:center;
            background: rgba(255,255,255,0.92);
            border-top: 1px solid rgba(0,0,0,0.08);
            z-index: 9999;
          }
          .block-container{ padding-bottom:70px !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def footer():
    st.markdown('<div class="ifaz-footer">Ifaz Ahmed Chowdhury &copy; 2026</div>', unsafe_allow_html=True)


def debug_expander(name: str, df: pd.DataFrame):
    with st.expander(f"Debug: {name} columns + sample", expanded=False):
        st.write("Columns:", list(df.columns))
        st.dataframe(df.head(10), use_container_width=True)


def module_complaints(acc_df: pd.DataFrame):
    st.header("Complaint Intelligence")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total complaints", len(acc_df))
    c2.metric("High risk", int((acc_df["risk_label"] == "High").sum()))
    c3.metric("Medium risk", int((acc_df["risk_label"] == "Medium").sum()))
    c4.metric("Low risk", int((acc_df["risk_label"] == "Low").sum()))

    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("Risk distribution")
        st.bar_chart(df_value_counts(acc_df["risk_label"], limit=None))
    with right:
        st.subheader("Amount bands")
        st.bar_chart(df_value_counts(acc_df["amount_band"], limit=None))

    st.subheader("Complaints over time")
    if acc_df["date"].notna().any():
        ts = (
            acc_df.dropna(subset=["date"])
            .assign(day=lambda d: d["date"].dt.date)
            .groupby("day")
            .size()
        )
        st.line_chart(ts)
    else:
        st.info("No valid 'date' values available for time series chart.")

    st.divider()

    st.subheader("Filters")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        division = st.selectbox("Division", ["All"] + sorted(acc_df["division"].unique().tolist()))
    with f2:
        sector = st.selectbox("Sector", ["All"] + sorted(acc_df["sector"].unique().tolist()))
    with f3:
        channel = st.selectbox("Channel", ["All"] + sorted(acc_df["channel"].unique().tolist()))
    with f4:
        amount_band = st.selectbox("Amount band", ["All"] + sorted(acc_df["amount_band"].unique().tolist()))

    filtered = acc_df.copy()
    if division != "All":
        filtered = filtered[filtered["division"] == division]
    if sector != "All":
        filtered = filtered[filtered["sector"] == sector]
    if channel != "All":
        filtered = filtered[filtered["channel"] == channel]
    if amount_band != "All":
        filtered = filtered[filtered["amount_band"] == amount_band]

    st.caption(f"Showing {len(filtered)} complaints after filters.")

    st.subheader("Top sectors (filtered)")
    st.bar_chart(df_value_counts(filtered["sector"], limit=12))

    st.subheader("Filtered table")
    show_cols = [
        "complaint_id","date","division","district","channel","sector","accused_type",
        "amount","amount_band","risk_label","complaint_text"
    ]
    show_cols = [c for c in show_cols if c in filtered.columns]
    filtered = filtered.sort_values("date", ascending=False) if "date" in filtered.columns else filtered
    st.dataframe(filtered[show_cols], use_container_width=True, height=340)

    st.divider()

    st.subheader("Risk prediction model (baseline)")
    with st.spinner("Training model..."):
        model, report = train_risk_model(acc_df)

    rows = []
    for label, metrics in report.items():
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
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=220)

    st.subheader("Inspect one complaint + prediction")
    pick_pool = filtered if len(filtered) else acc_df
    selected_id = st.selectbox("Select complaint_id", options=pick_pool["complaint_id"].astype(str).tolist())
    row = acc_df[acc_df["complaint_id"].astype(str) == str(selected_id)].iloc[0]

    X_one = pd.DataFrame([{
        "sector": row["sector"],
        "accused_type": row["accused_type"],
        "channel": row["channel"],
        "division": row["division"],
        "amount_band": row["amount_band"],
        "amount": row["amount"],
        "amount_log": row["amount_log"],
        "text_length": row["text_length"],
        "word_count": row["word_count"],
    }])

    pred = model.predict(X_one)[0]
    proba = model.predict_proba(X_one)[0]
    proba_tbl = pd.DataFrame({"label": list(model.classes_), "probability": [float(x) for x in proba]}).sort_values("probability", ascending=False)

    st.write(f"True label: **{row['risk_label']}**")
    st.write(f"Predicted: **{pred}**")
    st.dataframe(proba_tbl, use_container_width=True, height=160)
    st.write("Complaint text:")
    st.write(row.get("complaint_text", ""))

    debug_expander("ACC_COMPLAINTS", acc_df)


def module_procurement(proc_df: pd.DataFrame):
    st.header("Procurement Fraud Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total tenders", len(proc_df))
    c2.metric("Flagged (fraud_flag=1)", int(proc_df["fraud_flag"].sum()))
    c3.metric("Single bidder", int(proc_df["is_single_bidder"].sum()))
    c4.metric("High inflation", int(proc_df["is_high_inflation"].sum()))

    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("Suspicious signals (count)")
        signals = pd.Series({
            "fraud_flag=1": int(proc_df["fraud_flag"].sum()),
            "single bidder": int(proc_df["is_single_bidder"].sum()),
            "high inflation": int(proc_df["is_high_inflation"].sum()),
            "any rule hit": int((proc_df["rule_risk_score"] > 0).sum()),
        })
        st.bar_chart(signals)
    with right:
        st.subheader("Bidders count distribution")
        st.bar_chart(df_value_counts(proc_df["bidders_count"], limit=None))

    st.subheader("Top sectors by flagged count")
    by_sector = proc_df.groupby("sector")["fraud_flag"].sum().sort_values(ascending=False).head(12)
    st.bar_chart(by_sector)

    st.subheader("Inflation ratio buckets (contract / estimate)")
    ir = proc_df["inflation_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(ir):
        bins = pd.cut(ir.clip(upper=5), bins=[0, 0.8, 1.0, 1.1, 1.25, 1.5, 2.0, 5.0], include_lowest=True)
        st.bar_chart(bins.value_counts().sort_index())
    else:
        st.info("No inflation_ratio values available.")

    st.divider()

    st.subheader("Filters")
    f1, f2, f3 = st.columns(3)
    with f1:
        entity = st.selectbox("Procuring entity", ["All"] + sorted(proc_df["procuring_entity"].unique().tolist()))
    with f2:
        sector = st.selectbox("Sector", ["All"] + sorted(proc_df["sector"].unique().tolist()))
    with f3:
        method = st.selectbox("Method", ["All"] + sorted(proc_df["method"].unique().tolist()))
    only_suspicious = st.checkbox("Show only suspicious (fraud_flag=1 OR rule_risk_score>0)", value=True)

    df = proc_df.copy()
    if entity != "All":
        df = df[df["procuring_entity"] == entity]
    if sector != "All":
        df = df[df["sector"] == sector]
    if method != "All":
        df = df[df["method"] == method]
    if only_suspicious:
        df = df[(df["fraud_flag"] == 1) | (df["rule_risk_score"] > 0)]

    st.caption(f"Showing {len(df)} tenders after filters.")

    show_cols = [
        "tender_id","award_date","procuring_entity","supplier","method",
        "estimated_value","contract_value","bidders_count","sector","district",
        "fraud_flag","inflation_ratio","is_single_bidder","is_high_inflation","rule_risk_score"
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    if "award_date" in df.columns:
        df = df.sort_values("award_date", ascending=False)
    st.dataframe(df[show_cols], use_container_width=True, height=380)

    debug_expander("PROCUREMENT_FRAUD", proc_df)


def module_evidence(ev_df: pd.DataFrame):
    st.header("Evidence Viewer (OCR-like dataset)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total documents", len(ev_df))
    c2.metric("Overwritten flags", int(ev_df["flag_overwritten"].sum()))
    c3.metric("Mismatch total flags", int(ev_df["flag_mismatch_total"].sum()))
    c4.metric("Duplicate invoice flags", int(ev_df["flag_duplicate_invoice"].sum()))

    st.divider()

    st.subheader("Flag distribution (count)")
    flags = pd.Series({
        "overwritten": int(ev_df["flag_overwritten"].sum()),
        "mismatch_total": int(ev_df["flag_mismatch_total"].sum()),
        "duplicate_invoice": int(ev_df["flag_duplicate_invoice"].sum()),
        "impossible_date": int(ev_df["flag_impossible_date"].sum()),
    })
    st.bar_chart(flags)

    if ev_df["doc_type"].astype(str).str.strip().replace("", np.nan).notna().any():
        st.subheader("Document type distribution")
        st.bar_chart(df_value_counts(ev_df["doc_type"], limit=12))

    st.divider()

    st.subheader("Browse a document")
    doc_id = st.selectbox("Select sample_id", options=ev_df["sample_id"].astype(str).tolist())
    row = ev_df[ev_df["sample_id"].astype(str) == str(doc_id)].iloc[0]

    left, right = st.columns([1, 1])
    with left:
        st.write("Issuing entity:", row.get("issuing_entity", ""))
        st.write("Doc type:", row.get("doc_type", ""))
        if str(row.get("date_raw", "")).strip():
            st.write("Date:", row.get("date_raw", ""))

        st.subheader("Binary flags")
        st.json({
            "flag_overwritten": bool(int(row.get("flag_overwritten", 0))),
            "flag_mismatch_total": bool(int(row.get("flag_mismatch_total", 0))),
            "flag_duplicate_invoice": bool(int(row.get("flag_duplicate_invoice", 0))),
            "flag_impossible_date": bool(int(row.get("flag_impossible_date", 0))),
        })

        if str(row.get("red_flag", "")).strip():
            st.subheader("Red flag summary")
            st.write(row.get("red_flag", ""))

    with right:
        st.subheader("Raw OCR text")
        st.text(str(row.get("raw_text", ""))[:20000])

    debug_expander("EVIDENCE_OCR_DATASET", ev_df)


def module_rules(fraud_rules):
    st.header("Fraud Rules Library")

    if not isinstance(fraud_rules, list):
        st.warning("FRAUD_PATTERNS.json is not a list. Showing raw JSON.")
        st.json(fraud_rules)
        return

    df = pd.DataFrame([r for r in fraud_rules if isinstance(r, dict)])
    if df.empty:
        st.info("No rules found.")
        return

    st.subheader("Severity distribution")
    if "severity" in df.columns:
        st.bar_chart(df_value_counts(df["severity"], limit=None))
    else:
        st.info("No 'severity' field in rules dataset.")

    st.subheader("Category distribution")
    if "category" in df.columns:
        st.bar_chart(df_value_counts(df["category"], limit=12))
    else:
        st.info("No 'category' field in rules dataset.")

    st.divider()

    search = st.text_input("Search rules", "")
    filtered = df.copy()
    if search.strip():
        needle = search.strip().lower()
        mask = pd.Series(False, index=filtered.index)
        for col in filtered.columns:
            mask = mask | filtered[col].astype(str).str.lower().str.contains(needle, na=False)
        filtered = filtered[mask]

    if "category" in filtered.columns:
        cats = sorted(filtered["category"].astype(str).fillna("Uncategorized").unique().tolist())
        selected = st.selectbox("Filter by category", ["All"] + cats)
        if selected != "All":
            filtered = filtered[filtered["category"].astype(str) == selected]

    st.caption(f"Showing {len(filtered)} rules.")
    st.dataframe(filtered, use_container_width=True, height=420)

    with st.expander("Raw JSON"):
        st.json(fraud_rules)

    debug_expander("FRAUD_PATTERNS", df)


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="collapsed")
    inject_css()

    st.title(APP_TITLE)
    st.caption("Built on synthetic datasets for interview demonstration.")

    try:
        acc_df, proc_df, ev_df, fraud_rules = load_data()
    except Exception as e:
        st.error(str(e))
        st.stop()

    page = st.selectbox(
        "Select module",
        [
            "Complaint Intelligence",
            "Procurement Fraud Overview",
            "Evidence Viewer",
            "Fraud Rules Library",
        ],
    )

    st.divider()

    if page == "Complaint Intelligence":
        module_complaints(acc_df)
    elif page == "Procurement Fraud Overview":
        module_procurement(proc_df)
    elif page == "Evidence Viewer":
        module_evidence(ev_df)
    else:
        module_rules(fraud_rules)

    footer()


if __name__ == "__main__":
    main()
