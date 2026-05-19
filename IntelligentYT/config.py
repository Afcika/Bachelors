import os
from dotenv import load_dotenv

load_dotenv()

# Folders
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# Create folders if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("✅ Configuration loaded successfully!")
print(f"Data will be saved in: {DATA_DIR}")