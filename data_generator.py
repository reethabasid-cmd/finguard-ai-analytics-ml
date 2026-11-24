import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _normalize_probs(weights):
    arr = np.array(weights, dtype=float)
    return arr / arr.sum()


def generate_transactions(rows=10000, seed=42, inject_flaws=True):
    if rows < 100:
        raise ValueError("rows must be at least 100")

    rng = np.random.default_rng(seed)
    customer_count = max(500, rows // 5)
    customer_ids = rng.integers(1, customer_count + 1, size=rows)

    category_names = [
        "grocery", "electronics", "travel", "fashion",
        "digital_goods", "fuel", "restaurants",
    ]
    categories = rng.choice(
        category_names,
        rows,
        p=_normalize_probs([0.22, 0.14, 0.11, 0.17, 0.08, 0.13, 0.15]),
    )
    regions = rng.choice(
        ["North America", "Europe", "Asia Pacific", "Latin America"],
        rows,
        p=_normalize_probs([0.42, 0.26, 0.22, 0.10]),
    )
    payment_methods = rng.choice(
        ["card", "digital_wallet", "bank_transfer", "buy_now_pay_later"],
        rows,
        p=_normalize_probs([0.56, 0.24, 0.12, 0.08]),
    )

    timestamps = (
        pd.Timestamp.now(tz="UTC").floor("min")
        - pd.to_timedelta(rng.integers(0, 180, size=rows), unit="D")
        - pd.to_timedelta(rng.integers(0, 24 * 60, size=rows), unit="min")
    )
    account_age = np.clip(rng.gamma(shape=3.0, scale=350.0, size=rows), 1, 3650)
    frequency = np.clip(rng.poisson(lam=2.2, size=rows), 0, 25)
    device_risk = rng.beta(a=1.6, b=8.0, size=rows)

    category_multipliers = pd.Series(categories).map({
        "grocery": 0.7,
        "electronics": 2.7,
        "travel": 3.0,
        "fashion": 1.2,
        "digital_goods": 1.1,
        "fuel": 0.6,
        "restaurants": 0.8,
    }).to_numpy()
    amount = rng.lognormal(mean=3.45, sigma=0.85, size=rows) * category_multipliers
    amount = np.clip(amount, 2.0, 8000.0)

    logit = (
        -4.25
        + 2.7 * device_risk
        + 0.22 * np.maximum(frequency - 4, 0)
        + 0.9 * (account_age < 45)
        + 0.65 * (amount > 900)
        + 0.45 * np.isin(categories, ["digital_goods", "electronics"])
        + 0.55 * np.isin(payment_methods, ["digital_wallet", "buy_now_pay_later"])
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    is_fraud = rng.binomial(1, probability)

    fraud_count = max(1, round(rows * 0.04))
    if int(is_fraud.sum()) > fraud_count:
        fraud_indices = np.flatnonzero(is_fraud)
        keep = rng.choice(fraud_indices, size=fraud_count, replace=False)
        is_fraud[:] = 0
        is_fraud[keep] = 1
    elif int(is_fraud.sum()) < fraud_count:
        eligible = np.flatnonzero(is_fraud == 0)
        extra = rng.choice(
            eligible, size=fraud_count - int(is_fraud.sum()), replace=False
        )
        is_fraud[extra] = 1

    fraud_mask = is_fraud == 1
    device_risk[fraud_mask] = np.maximum(
        device_risk[fraud_mask], rng.uniform(0.55, 0.98, fraud_mask.sum())
    )
    frequency[fraud_mask] = np.maximum(
        frequency[fraud_mask], rng.integers(5, 15, fraud_mask.sum())
    )
    account_age[fraud_mask] = np.minimum(
        account_age[fraud_mask], rng.integers(1, 120, fraud_mask.sum())
    )
    amount[fraud_mask] *= rng.uniform(1.3, 3.2, fraud_mask.sum())

    frame = pd.DataFrame({
        "transaction_id": [f"TX-{i:08d}" for i in range(1, rows + 1)],
        "timestamp": timestamps.astype(str),
        "customer_id": [f"C-{value:06d}" for value in customer_ids],
        "merchant_category": categories,
        "region": regions,
        "payment_method": payment_methods,
        "amount": np.round(amount, 2),
        "account_age_days": np.round(account_age).astype(int),
        "tx_frequency_24h": frequency.astype(int),
        "device_risk_score": np.round(device_risk, 4),
        "is_fraud": is_fraud.astype(int),
    })

    if inject_flaws:
        for column, rate in {
            "merchant_category": 0.006,
            "region": 0.004,
            "amount": 0.004,
            "account_age_days": 0.003,
            "device_risk_score": 0.003,
        }.items():
            indices = rng.choice(rows, size=max(1, round(rows * rate)), replace=False)
            frame.loc[indices, column] = np.nan

        negative_indices = rng.choice(rows, size=max(1, round(rows * 0.002)), replace=False)
        frame.loc[negative_indices, "amount"] = -frame.loc[negative_indices, "amount"].abs()

        duplicate_count = max(1, round(rows * 0.005))
        duplicates = frame.sample(n=duplicate_count, random_state=seed)
        frame = pd.concat([frame, duplicates], ignore_index=True)

    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/transactions.csv"))
    parser.add_argument("--no-flaws", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = generate_transactions(
        rows=args.rows, seed=args.seed, inject_flaws=not args.no_flaws
    )
    data.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
