"""Download the original UCI Online Shoppers Purchasing Intention dataset."""
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "online_shoppers_intention.csv"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        data = pd.read_csv(OUT)
        print(f"Dataset already available: {OUT}")
    else:
        print("Fetching UCI dataset #468 (internet required)...")
        dataset = fetch_ucirepo(id=468)
        x = dataset.data.features.reset_index(drop=True)
        y = dataset.data.targets.reset_index(drop=True)
        if "Revenue" not in y.columns:
            raise ValueError(f"Expected Revenue column, got {y.columns.tolist()}")
        data = pd.concat([x, y[["Revenue"]]], axis=1)
        data.to_csv(OUT, index=False)
        print(f"Saved dataset: {OUT}")

    if "Revenue" not in data:
        raise ValueError("Missing target column 'Revenue'.")
    print(f"Rows: {len(data):,} | Columns: {len(data.columns)}")
    print(f"Features: {len(data.columns)-1} | Target: Revenue")


if __name__ == "__main__":
    main()
