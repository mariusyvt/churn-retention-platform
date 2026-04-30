

from __future__ import annotations

import os
import sys


if __package__ is None or __package__ == "":
    project_root = os.path.dirname(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.data.loader import audit_dataset, load_data
from src.data.preprocessor import prepare_data, save_preprocessor


DATA_PATH = os.path.join("data", "raw", "customer_churn.csv")
PREPROCESSOR_PATH = os.path.join("models", "preprocessor.pkl")


def main() -> None:
    print("=" * 50)
    print("Chargement et execution du preprocessor")
    print("=" * 50)

    df = load_data(DATA_PATH)
    audit_dataset(df)
    _, _, _, _, preprocessor, feature_names = prepare_data(df)
    save_preprocessor(preprocessor, PREPROCESSOR_PATH)

    print(f"\Preprocessor genere : {PREPROCESSOR_PATH}")
    print(f"Nombre de features apres preparation : {len(feature_names)}")


if __name__ == "__main__":
    main()


