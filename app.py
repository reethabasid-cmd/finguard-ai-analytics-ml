"""FinGuard AI executive dashboard and transaction risk copilot."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import plotly.express as px
import streamlit as st

from data_generator import generate_transactions
from etl_and_features import clean_transactions, engineer_features
from ml_engine import ARTIFACT_PATH, load_artifacts, score_batch, score_transaction, train_and_save

st.set_page_config(
    page_title="FinGuard AI | Revenue & Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
)

DATA_PATH = Path("data/transactions.csv")


@st.cache_resource
def bootstrap() -> tuple[pd.DataFrame, dict, dict]:
    """Create demo data and trained artifacts on first launch."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        generate_transactions().to_csv(DATA_PATH, index=False)
    raw = pd.read_csv(DATA_PATH)
    clean, report = clean_transactions(raw)
    if not ARTIFACT_PATH.exists():
        artifacts = train_and_save(raw)
    else:
        artifacts = load_artifacts()
    # Use current ETL report for the dashboard even when artifacts already exist.
    return clean, report, artifacts


try:
    clean_data, etl_report, artifacts = bootstrap()
except Exception as exc:
    st.error(f"Could not initialize FinGuard AI: {exc}")
    st.stop()

featured_data = engineer_features(clean_data, artifacts["baseline_amount"])
st.title("🛡️ FinGuard AI")
st.caption("Enterprise Revenue Analytics • Fraud Detection • GenAI Executive Copilot")

tabs = st.tabs([
    "Executive Analytics & ETL Health",
    "ML Performance & Explainability",
    "Live Transaction Risk Scorer",
    "Batch Scoring & AI Executive Brief",
])

with tabs[0]:
    st.subheader("Executive analytics")
    total_volume = float(clean_data["amount"].sum())
    fraud_rate = float(clean_data["is_fraud"].mean()) if "is_fraud" in clean_data else 0.0
    prevented_loss = float(
        clean_data.loc[clean_data.get("is_fraud", 0) == 1, "amount"].sum()
    ) if "is_fraud" in clean_data else 0.0

    kpis = st.columns(4)
    kpis[0].metric("Total volume", f"${total_volume:,.0f}")
    kpis[1].metric("Cleaned records", f"{etl_report['cleaned_record_pct']:.1f}%")
    kpis[2].metric("Fraud rate", f"{fraud_rate:.2%}")
    kpis[3].metric("Fraud exposure", f"${prevented_loss:,.0f}")

    st.markdown("#### ETL pipeline audit")
    audit_cols = st.columns(4)
    audit_cols[0].metric("Rows ingested", f"{etl_report['rows_ingested']:,}")
    audit_cols[1].metric("Rows cleaned", f"{etl_report['rows_cleaned']:,}")
    audit_cols[2].metric("Duplicates removed", f"{etl_report['duplicates_removed']:,}")
    audit_cols[3].metric("Anomalies fixed", f"{etl_report['anomalies_fixed']:,}")
    with st.expander("Full data-quality report"):
        st.json(etl_report)

    chart_frame = clean_data.copy()
    chart_frame["date"] = chart_frame["timestamp"].dt.floor("D")
    daily = chart_frame.groupby("date", as_index=False).agg(
        revenue=("amount", "sum"),
        fraud_count=("is_fraud", "sum") if "is_fraud" in chart_frame else ("amount", "count"),
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.line(daily, x="date", y="revenue", title="Revenue trend",
                    labels={"date": "Date", "revenue": "Revenue ($)"}),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            px.bar(daily, x="date", y="fraud_count", title="Fraud transactions over time",
                   labels={"date": "Date", "fraud_count": "Fraud transactions"}),
            use_container_width=True,
        )

    group_col = st.selectbox("Risk breakdown", ["region", "merchant_category"])
    risk = chart_frame.groupby(group_col, as_index=False).agg(
        transactions=("amount", "size"),
        fraud_rate=("is_fraud", "mean") if "is_fraud" in chart_frame else ("amount", "mean"),
    )
    st.plotly_chart(
        px.bar(risk, x=group_col, y="fraud_rate", color="transactions",
               title=f"Fraud rate by {group_col.replace('_', ' ')}",
               labels={"fraud_rate": "Fraud rate", group_col: group_col.title()}),
        use_container_width=True,
    )

with tabs[1]:
    st.subheader(f"Champion model: {artifacts['model_name']}")
    st.caption("Evaluation uses a stratified holdout set. The 0.50 decision threshold is configurable in code.")
    metric_cols = st.columns(4)
    for col, key, label in zip(
        metric_cols, ["roc_auc", "precision", "recall", "f1"],
        ["ROC-AUC", "Precision", "Recall", "F1-score"],
    ):
        col.metric(label, f"{artifacts['metrics'][key]:.3f}")

    left, right = st.columns(2)
    matrix = artifacts["metrics"]["confusion_matrix"]
    with left:
        st.plotly_chart(
            px.imshow(matrix, text_auto=True, x=["Predicted 0", "Predicted 1"],
                      y=["Actual 0", "Actual 1"], color_continuous_scale="Blues",
                      title="Confusion matrix"),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            px.bar(pd.DataFrame(artifacts["feature_importances"][:15]),
                   x="importance", y="feature", orientation="h",
                   title="Top global feature importances"),
            use_container_width=True,
        )
    st.markdown("#### Model comparison")
    comparison = pd.DataFrame(artifacts["all_model_metrics"]).T
    st.dataframe(comparison[["roc_auc", "precision", "recall", "f1"]].style.format("{:.3f}"),
                 use_container_width=True)
    st.caption("Tree feature importances summarize global model usage; they are not causal explanations.")

with tabs[2]:
    st.subheader("Score a transaction")
    with st.form("risk_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            amount = st.number_input("Amount ($)", min_value=0.01, value=250.0, step=10.0)
            category = st.selectbox("Merchant category", sorted(clean_data["merchant_category"].unique()))
            region = st.selectbox("Region", sorted(clean_data["region"].unique()))
        with c2:
            method = st.selectbox("Payment method", sorted(clean_data["payment_method"].unique()))
            age = st.number_input("Account age (days)", min_value=0, value=365, step=30)
            frequency = st.slider("Transactions in last 24 hours", 0, 30, 2)
        with c3:
            device_risk = st.slider("Device risk score", 0.0, 1.0, 0.15, 0.01)
            customer_id = st.text_input("Customer ID", "LIVE-CUSTOMER")
            timestamp = st.date_input("Transaction date", value=pd.Timestamp.now().date())
        submitted = st.form_submit_button("Assess transaction risk", type="primary")

    if submitted:
        result = score_transaction({
            "transaction_id": "LIVE-SIMULATION",
            "timestamp": pd.Timestamp(timestamp, tz="UTC").isoformat(),
            "customer_id": customer_id,
            "merchant_category": category,
            "region": region,
            "payment_method": method,
            "amount": amount,
            "account_age_days": age,
            "tx_frequency_24h": frequency,
            "device_risk_score": device_risk,
        }, artifacts)
        a, b = st.columns(2)
        a.metric("Fraud probability", f"{result['fraud_probability']:.1%}")
        b.metric("Risk tier", result["risk_tier"])
        st.markdown("**Top model-informed risk indicators**")
        for reason in result["top_risk_factors"]:
            st.write(f"• {reason}")
        st.caption("Indicators are importance-ranked model features, not causal attributions.")

with tabs[3]:
    st.subheader("Batch scoring")
    uploaded = st.file_uploader("Upload transaction CSV", type=["csv"])
    if uploaded is not None:
        try:
            upload_frame = pd.read_csv(uploaded)
            scored = score_batch(upload_frame, artifacts)
            st.success(f"Scored {len(scored):,} cleaned transactions.")
            st.dataframe(scored.head(100), use_container_width=True)
            st.download_button(
                "Download scored CSV",
                scored.to_csv(index=False).encode("utf-8"),
                file_name="finguard_scored_transactions.csv",
                mime="text/csv",
            )
            st.markdown("#### AI executive brief")
            summary = (
                f"Scored {len(scored):,} transactions. "
                f"{int((scored['risk_tier'] == 'Critical').sum()):,} are Critical and "
                f"{int((scored['risk_tier'] == 'Medium').sum()):,} are Medium risk. "
                f"Total transaction value is ${scored['amount'].sum():,.0f}; "
                f"average estimated fraud probability is "
                f"{scored['fraud_probability'].mean():.2%}."
            )
            api_key = st.text_input(
                "Optional OpenAI API key (used for this request only)",
                type="password",
                help="Leave blank to use the built-in free rule-based summary.",
            )
            if st.button("Generate executive brief"):
                if api_key:
                    try:
                        payload = {
                            "model": "gpt-4o-mini",
                            "messages": [
                                {"role": "system", "content": "Write a concise, cautious executive fraud-risk brief. Do not claim causality."},
                                {"role": "user", "content": f"Summarize these batch results: {summary}"},
                            ],
                            "temperature": 0.2,
                        }
                        request = Request(
                            "https://api.openai.com/v1/chat/completions",
                            data=json.dumps(payload).encode(),
                            headers={
                                "Authorization": f"Bearer {api_key}",
                                "Content-Type": "application/json",
                            },
                            method="POST",
                        )
                        with urlopen(request, timeout=20) as response:
                            body = json.loads(response.read().decode())
                        summary = body["choices"][0]["message"]["content"]
                    except Exception as exc:
                        st.warning(f"OpenAI request failed; showing the built-in summary instead. ({exc})")
                st.info(summary)
        except Exception as exc:
            st.error(f"Unable to score this CSV: {exc}")
            st.caption(
                "The uploaded file should contain transaction columns such as amount, "
                "merchant_category, region, payment_method, account_age_days, "
                "tx_frequency_24h, and device_risk_score. Missing fields are imputed."
            )

st.divider()
st.caption(
    "FinGuard AI is a portfolio demonstration, not a production payment decision system. "
    "Validate calibration, fairness, privacy, and operational controls before real-world use."
)
