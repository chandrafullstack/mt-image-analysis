import pandas as pd
from pathlib import Path

base = Path("c:/Chandra/MT/mito_classifier/outputs/models_v2_round1")
for s in ["train", "val", "test"]:
    d = pd.read_csv(base / f"manifest_{s}.csv")
    print(f"=== {s} ===")
    print(f"  rows: {len(d)}, source_files: {d['source_file'].nunique()}")
    print(f"  label dist: {dict(d['label'].value_counts())}")
