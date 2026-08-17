"""Anonymize grocery retailer data by replacing identifying columns with sequential IDs.

Usage:
    python scenarios/scenario_grocery_retailer/scripts/anonymize_data.py [--input data/] [--output data/anonymized/]

Replaces: ARTIKELNR, KUNDENNR, PERS_NR, AUFTRAGSNR with sequential integers.
Preserves: all timing, quantity, weight, volume, and location data.
The layout file is not modified (contains only structural layout codes).
"""
import argparse
from pathlib import Path

import pandas as pd


def anonymize_orders(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path, sep=";")

    mappings = {}
    for col in ["ARTIKELNR", "KUNDENNR", "PERS_NR", "AUFTRAGSNR"]:
        unique_vals = sorted(df[col].unique())
        mappings[col] = {val: i + 1 for i, val in enumerate(unique_vals)}
        df[col] = df[col].map(mappings[col])

    df.to_csv(output_path, sep=";", index=False)
    print(f"Anonymized {len(df)} rows -> {output_path}")
    print(f"  ARTIKELNR: {len(mappings['ARTIKELNR'])} unique -> 1..{len(mappings['ARTIKELNR'])}")
    print(f"  KUNDENNR:  {len(mappings['KUNDENNR'])} unique -> 1..{len(mappings['KUNDENNR'])}")
    print(f"  PERS_NR:   {len(mappings['PERS_NR'])} unique -> 1..{len(mappings['PERS_NR'])}")
    print(f"  AUFTRAGSNR: {len(mappings['AUFTRAGSNR'])} unique -> 1..{len(mappings['AUFTRAGSNR'])}")


def copy_layout(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path, sep=";")
    df.to_csv(output_path, sep=";", index=False)
    print(f"Copied layout ({df.shape}) -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Anonymize grocery retailer data")
    parser.add_argument("--input", type=Path, default=Path("scenarios/scenario_grocery_retailer/data"), help="Input data directory")
    parser.add_argument("--output", type=Path, default=Path("scenarios/scenario_grocery_retailer/data/anonymized"), help="Output directory")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    anonymize_orders(args.input / "orders.csv", args.output / "orders.csv")
    copy_layout(args.input / "layout.csv", args.output / "layout.csv")
    print(f"\nDone. Anonymized data in {args.output}/")


if __name__ == "__main__":
    main()
