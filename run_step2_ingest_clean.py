from src.utils.paths import ensure_dirs
from src.data_ingestion.ingest import main as ingest_main
from src.data_cleaning.clean import main as clean_main

if __name__ == "__main__":
    ensure_dirs()
    print("=== STEP 2: Ingestion ===")
    ingest_main()
    print("\n=== STEP 2: Cleaning ===")
    clean_main()
    print("\nAll good. You can inspect data/processed/*.csv now.")
