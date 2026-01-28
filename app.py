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

# --- কনফিগারেশন এবং বাংলা টাইটেল ---
APP_TITLE = "দুদক (ACC) দুর্নীতি বিরোধী অ্যানালিটিক্স ডেমো"

# --- বাংলা ম্যাপিং ডিকশনারি (চার্ট ও ডিসপ্লের জন্য) ---
RISK_MAP = {"High": "উচ্চ", "Medium": "মাঝারি", "Low": "নিম্ন", "Very High": "খুব উচ্চ"}
FRAUD_MAP = {0: "স্বাভাবিক", 1: "সন্দেহজনক"}

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


# FIX: cache_resource -> cache_data for better compatibility with dictionary returns and pickling
@st.cache_data(show_spinner=False)
def train_risk_model(df: pd.DataFrame):
    # নোট: এখানে পুনরায় enrich_complaints কল করা অপ্রয়োজনীয় কারণ load_data তে একবার করা হয়েছে।
    
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

    # FIX: Explicit solver to avoid version conflicts, though default is usually fine
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", LogisticRegression(max_iter=2000, multi_class="auto", solver='lbfgs')),
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
        top.loc["অন্যান্য (Other)"] = int(vc.iloc[limit:].sum())
        return top
    return vc


def inject_css():
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Hind+Siliguri:wght@300;400;600&display=swap');
          html, body, [class*="css"] {
            font-family: 'Hind Siliguri', sans-serif;
          }
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
    with st.expander(f"ডিবাগ: {name} কলাম + স্যাম্পল", expanded=False):
        st.write("কলামসমূহ:", list(df.columns))
        st.dataframe(df.head(10), use_container_width=True)


def module_complaints(acc_df: pd.DataFrame):
    st.header("অভিযোগ ইন্টেলিজেন্স (Complaint Intelligence)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("মোট অভিযোগ", len(acc_df))
    c2.metric("উচ্চ ঝুঁকি (High)", int((acc_df["risk_label"] == "High").sum()))
    c3.metric("মাঝারি ঝুঁকি (Medium)", int((acc_df["risk_label"] == "Medium").sum()))
    c4.metric("নিম্ন ঝুঁকি (Low)", int((acc_df["risk_label"] == "Low").sum()))

    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("ঝুঁকির বণ্টন")
        risk_counts = df_value_counts(acc_df["risk_label"], limit=None)
        risk_counts.index = risk_counts.index.map(lambda x: RISK_MAP.get(x, x))
        st.bar_chart(risk_counts)
    with right:
        st.subheader("টাকার পরিমাণ (ব্যান্ড)")
        st.bar_chart(df_value_counts(acc_df["amount_band"], limit=None))

    st.subheader("সময়ের সাথে অভিযোগের ধারা")
    if acc_df["date"].notna().any():
        ts = (
            acc_df.dropna(subset=["date"])
            .assign(day=lambda d: d["date"].dt.date)
            .groupby("day")
            .size()
        )
        st.line_chart(ts)
    else:
        st.info("চার্টের জন্য পর্যাপ্ত তারিখের তথ্য পাওয়া যায়নি।")

    st.divider()

    st.subheader("ফিল্টার অপশন")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        division = st.selectbox("বিভাগ", ["সব (All)"] + sorted(acc_df["division"].unique().tolist()))
    with f2:
        sector = st.selectbox("খাত (Sector)", ["সব (All)"] + sorted(acc_df["sector"].unique().tolist()))
    with f3:
        channel = st.selectbox("মাধ্যম (Channel)", ["সব (All)"] + sorted(acc_df["channel"].unique().tolist()))
    with f4:
        amount_band = st.selectbox("টাকার ব্যান্ড", ["সব (All)"] + sorted(acc_df["amount_band"].unique().tolist()))

    filtered = acc_df.copy()
    if division != "সব (All)":
        filtered = filtered[filtered["division"] == division]
    if sector != "সব (All)":
        filtered = filtered[filtered["sector"] == sector]
    if channel != "সব (All)":
        filtered = filtered[filtered["channel"] == channel]
    if amount_band != "সব (All)":
        filtered = filtered[filtered["amount_band"] == amount_band]

    st.caption(f"ফিল্টার করার পর {len(filtered)} টি অভিযোগ পাওয়া গেছে।")

    st.subheader("শীর্ষ খাতসমূহ (ফিল্টার করা)")
    st.bar_chart(df_value_counts(filtered["sector"], limit=12))

    st.subheader("ফিল্টার করা তালিকা")
    show_cols = [
        "complaint_id","date","division","district","channel","sector","accused_type",
        "amount","risk_label","complaint_text"
    ]
    show_cols = [c for c in show_cols if c in filtered.columns]
    
    # ডিসপ্লের জন্য বাংলা কলাম নাম
    display_df = filtered[show_cols].copy()
    if "date" in display_df.columns:
        display_df = display_df.sort_values("date", ascending=False)
        
    st.dataframe(display_df, use_container_width=True, height=340)

    st.divider()

    st.subheader("ঝুঁকি পূর্বাভাস মডেল (AI Model)")
    with st.spinner("মডেল ট্রেনিং হচ্ছে, অনুগ্রহ করে অপেক্ষা করুন..."):
        model, report = train_risk_model(acc_df)

    rows = []
    for label, metrics in report.items():
        if label in ["accuracy", "macro avg", "weighted avg"]:
            continue
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "ঝুঁকি লেভেল": RISK_MAP.get(label, label),
                "প্রিসিশন (Precision)": round(metrics.get("precision", 0), 2),
                "রিকল (Recall)": round(metrics.get("recall", 0), 2),
                "এফ১ স্কোর (F1)": round(metrics.get("f1-score", 0), 2),
                "সংখ্যা (Support)": int(metrics.get("support", 0)),
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=220)

    st.subheader("একটি অভিযোগ যাচাই করুন")
    pick_pool = filtered if len(filtered) else acc_df
    selected_id = st.selectbox("অভিযোগ আইডি নির্বাচন করুন", options=pick_pool["complaint_id"].astype(str).tolist())
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
    
    # প্রোবাবিলিটি টেবিল
    proba_tbl = pd.DataFrame({
        "লেভেল": [RISK_MAP.get(c, c) for c in model.classes_], 
        "সম্ভাবনা": [f"{x*100:.1f}%" for x in proba]
    })

    st.write(f"আসল লেভেল: **{RISK_MAP.get(row['risk_label'], row['risk_label'])}**")
    st.write(f"মডেলের পূর্বাভাস: **{RISK_MAP.get(pred, pred)}**")
    st.dataframe(proba_tbl, use_container_width=True, height=160)
    st.write("অভিযোগের বিবরণ:")
    st.info(row.get("complaint_text", ""))

    debug_expander("ACC_COMPLAINTS", acc_df)


def module_procurement(proc_df: pd.DataFrame):
    st.header("ক্রয় দুর্নীতি ওভারভিউ (Procurement Fraud)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("মোট টেন্ডার", len(proc_df))
    c2.metric("চিহ্নিত ফ্রড", int(proc_df["fraud_flag"].sum()))
    c3.metric("একক বিডার", int(proc_df["is_single_bidder"].sum()))
    c4.metric("উচ্চ মূল্যস্ফীতি", int(proc_df["is_high_inflation"].sum()))

    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("সন্দেহজনক সংকেত (সংখ্যা)")
        signals = pd.Series({
            "ফ্রড ফ্ল্যাগ (Flagged)": int(proc_df["fraud_flag"].sum()),
            "একক বিডার (Single Bidder)": int(proc_df["is_single_bidder"].sum()),
            "উচ্চ মূল্যস্ফীতি (High Inflation)": int(proc_df["is_high_inflation"].sum()),
            "রুল ভায়োলেশন (Any Rule Hit)": int((proc_df["rule_risk_score"] > 0).sum()),
        })
        st.bar_chart(signals)
    with right:
        st.subheader("বিডার সংখ্যার বণ্টন")
        st.bar_chart(df_value_counts(proc_df["bidders_count"], limit=None))

    st.subheader("শীর্ষ খাত (ফ্রড ফ্ল্যাগ অনুযায়ী)")
    by_sector = proc_df.groupby("sector")["fraud_flag"].sum().sort_values(ascending=False).head(12)
    st.bar_chart(by_sector)

    st.subheader("মূল্যস্ফীতির অনুপাত (চুক্তি / প্রাক্কলন)")
    ir = proc_df["inflation_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(ir):
        bins = pd.cut(ir.clip(upper=5), bins=[0, 0.8, 1.0, 1.1, 1.25, 1.5, 2.0, 5.0], include_lowest=True)
        st.bar_chart(bins.value_counts().sort_index())
    else:
        st.info("মূল্যস্ফীতির অনুপাত পাওয়া যায়নি।")

    st.divider()

    st.subheader("ফিল্টার অপশন")
    f1, f2, f3 = st.columns(3)
    with f1:
        entity = st.selectbox("ক্রয়কারী প্রতিষ্ঠান", ["সব (All)"] + sorted(proc_df["procuring_entity"].unique().tolist()))
    with f2:
        sector = st.selectbox("খাত", ["সব (All)"] + sorted(proc_df["sector"].unique().tolist()))
    with f3:
        method = st.selectbox("পদ্ধতি (Method)", ["সব (All)"] + sorted(proc_df["method"].unique().tolist()))
    only_suspicious = st.checkbox("শুধুমাত্র সন্দেহজনক দেখান (Show only suspicious)", value=True)

    df = proc_df.copy()
    if entity != "সব (All)":
        df = df[df["procuring_entity"] == entity]
    if sector != "সব (All)":
        df = df[df["sector"] == sector]
    if method != "সব (All)":
        df = df[df["method"] == method]
    if only_suspicious:
        df = df[(df["fraud_flag"] == 1) | (df["rule_risk_score"] > 0)]

    st.caption(f"ফিল্টার করার পর {len(df)} টি টেন্ডার পাওয়া গেছে।")

    show_cols = [
        "tender_id","award_date","procuring_entity","supplier","method",
        "estimated_value","contract_value","bidders_count","sector","district",
        "fraud_flag","rule_risk_score"
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    if "award_date" in df.columns:
        df = df.sort_values("award_date", ascending=False)
    st.dataframe(df[show_cols], use_container_width=True, height=380)

    debug_expander("PROCUREMENT_FRAUD", proc_df)


def module_evidence(ev_df: pd.DataFrame):
    st.header("প্রমাণ ভিউয়ার (Evidence Viewer)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("মোট নথি", len(ev_df))
    c2.metric("ওভাররাইট ফ্ল্যাগ", int(ev_df["flag_overwritten"].sum()))
    c3.metric("অমিল ফ্ল্যাগ", int(ev_df["flag_mismatch_total"].sum()))
    c4.metric("ডুপ্লিকেট ইনভয়েস", int(ev_df["flag_duplicate_invoice"].sum()))

    st.divider()

    st.subheader("ফ্ল্যাগ ডিস্ট্রিবিউশন")
    flags = pd.Series({
        "কাটাকাটি/ওভাররাইট": int(ev_df["flag_overwritten"].sum()),
        "হিসাবে অমিল": int(ev_df["flag_mismatch_total"].sum()),
        "ডুপ্লিকেট ইনভয়েস": int(ev_df["flag_duplicate_invoice"].sum()),
        "অসম্ভব তারিখ": int(ev_df["flag_impossible_date"].sum()),
    })
    st.bar_chart(flags)

    if ev_df["doc_type"].astype(str).str.strip().replace("", np.nan).notna().any():
        st.subheader("নথির ধরন")
        st.bar_chart(df_value_counts(ev_df["doc_type"], limit=12))

    st.divider()

    st.subheader("নথি যাচাই করুন")
    doc_id = st.selectbox("স্যাম্পল আইডি নির্বাচন করুন", options=ev_df["sample_id"].astype(str).tolist())
    row = ev_df[ev_df["sample_id"].astype(str) == str(doc_id)].iloc[0]

    left, right = st.columns([1, 1])
    with left:
        st.write("ইস্যুকারী প্রতিষ্ঠান:", row.get("issuing_entity", ""))
        st.write("নথির ধরন:", row.get("doc_type", ""))
        if str(row.get("date_raw", "")).strip():
            st.write("তারিখ:", row.get("date_raw", ""))

        st.subheader("অটোমেটেড ফ্ল্যাগ")
        st.json({
            "কাটাকাটি আছে?": bool(int(row.get("flag_overwritten", 0))),
            "টাকার অংকে অমিল?": bool(int(row.get("flag_mismatch_total", 0))),
            "ডুপ্লিকেট?": bool(int(row.get("flag_duplicate_invoice", 0))),
            "তারিখ ভুল?": bool(int(row.get("flag_impossible_date", 0))),
        })

        if str(row.get("red_flag", "")).strip():
            st.subheader("রেড ফ্ল্যাগ সারাংশ")
            st.warning(row.get("red_flag", ""))

    with right:
        st.subheader("OCR টেক্সট (Raw)")
        st.text_area("টেক্সট কন্টেন্ট", str(row.get("raw_text", ""))[:20000], height=400)

    debug_expander("EVIDENCE_OCR_DATASET", ev_df)


def module_rules(fraud_rules):
    st.header("ফ্রড রুলস লাইব্রেরি")

    if not isinstance(fraud_rules, list):
        st.warning("FRAUD_PATTERNS.json সঠিক ফরম্যাটে নেই।")
        st.json(fraud_rules)
        return

    df = pd.DataFrame([r for r in fraud_rules if isinstance(r, dict)])
    if df.empty:
        st.info("কোনো রুল পাওয়া যায়নি।")
        return

    st.subheader("তীব্রতা অনুযায়ী রুলস (Severity)")
    if "severity" in df.columns:
        st.bar_chart(df_value_counts(df["severity"], limit=None))
    else:
        st.info("'severity' ফিল্ড পাওয়া যায়নি।")

    st.subheader("ক্যাটাগরি অনুযায়ী")
    if "category" in df.columns:
        st.bar_chart(df_value_counts(df["category"], limit=12))
    else:
        st.info("'category' ফিল্ড পাওয়া যায়নি।")

    st.divider()

    search = st.text_input("রুল খুঁজুন (Search Rules)", "")
    filtered = df.copy()
    if search.strip():
        needle = search.strip().lower()
        mask = pd.Series(False, index=filtered.index)
        for col in filtered.columns:
            mask = mask | filtered[col].astype(str).str.lower().str.contains(needle, na=False)
        filtered = filtered[mask]

    if "category" in filtered.columns:
        cats = sorted(filtered["category"].astype(str).fillna("Uncategorized").unique().tolist())
        selected = st.selectbox("ক্যাটাগরি ফিল্টার", ["সব (All)"] + cats)
        if selected != "সব (All)":
            filtered = filtered[filtered["category"].astype(str) == selected]

    st.caption(f"{len(filtered)} টি রুল দেখানো হচ্ছে।")
    st.dataframe(filtered, use_container_width=True, height=420)

    with st.expander("র 'জেসন' (Raw JSON) দেখুন"):
        st.json(fraud_rules)

    debug_expander("FRAUD_PATTERNS", df)


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="collapsed")
    inject_css()

    st.title(APP_TITLE)
    st.caption("ইন্টারভিউ ডেমোনস্ট্রেশনের জন্য সিনথেটিক ডেটাসেট দ্বারা তৈরি।")

    try:
        acc_df, proc_df, ev_df, fraud_rules = load_data()
    except Exception as e:
        st.error(f"ডেটা লোড করতে সমস্যা হয়েছে: {str(e)}")
        st.stop()

    page = st.selectbox(
        "মডিউল নির্বাচন করুন",
        [
            "অভিযোগ ইন্টেলিজেন্স (Complaint Intelligence)",
            "ক্রয় দুর্নীতি ওভারভিউ (Procurement Fraud)",
            "প্রমাণ ভিউয়ার (Evidence Viewer)",
            "ফ্রড রুলস লাইব্রেরি (Rules Library)",
        ],
    )

    st.divider()

    if "অভিযোগ" in page:
        module_complaints(acc_df)
    elif "ক্রয়" in page:
        module_procurement(proc_df)
    elif "প্রমাণ" in page:
        module_evidence(ev_df)
    else:
        module_rules(fraud_rules)

    footer()


if __name__ == "__main__":
    main()