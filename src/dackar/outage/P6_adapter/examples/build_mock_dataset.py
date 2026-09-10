from pathlib import Path

from outage_model.loaders.example_loader import load_mock_dataset


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent / "mock_p6_export"
    dataset = load_mock_dataset(base_dir)

    print("Loaded tables:")
    for table_name, df in dataset.as_dict().items():
        print(f"- {table_name}: {len(df)} rows")

    print("\nSample schedule tasks:")
    print(dataset.schedule_tasks.head().to_string(index=False))
