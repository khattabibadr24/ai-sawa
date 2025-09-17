import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.vector_db_service import initialize_vector_db_from_data

def init_vector_database():
    json_file_path = PROJECT_ROOT / "data" / "medical_data.json"
    initialize_vector_db_from_data(str(json_file_path))

def main():
    json_file_path = PROJECT_ROOT / "data" / "medical_data.json"
    
    if not json_file_path.exists():
        print(f"❌ Error: Medical data file not found at {json_file_path}")
        print("Please ensure the medical_data.json file exists in the data/ directory.")
        sys.exit(1)
    
    print("🚀 Initializing vector database...")
    print(f"📁 Data file: {json_file_path}")
    
    try:
        init_vector_database()
        print("✅ Vector database initialized successfully!")
        print("🎯 You can now start the application with: python main.py")
    except Exception as e:
        print(f"❌ Error initializing vector database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
