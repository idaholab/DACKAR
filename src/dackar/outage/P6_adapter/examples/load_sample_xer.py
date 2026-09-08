from pathlib import Path

from outage_model.loaders.xer_loader import load_xer_dataset


if __name__ == "__main__":
    xer_path = Path(__file__).resolve().parent / "sample_xer" / "sample_project.xer"
    dataset = load_xer_dataset(xer_path)
    print("Tables loaded:")
    for name, df in dataset.as_dict().items():
        if not df.empty:
            print(f"- {name}: {len(df)} rows")
