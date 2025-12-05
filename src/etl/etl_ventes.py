from src.etl.preprocess_ventes import preprocess_ventes

def run_ventes_etl():
    print("=== Running ETL VENTES ===")
    preprocess_ventes()
    print("ETL Completed!")

if __name__ == "__main__":
    run_ventes_etl()
