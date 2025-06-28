import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment
load_dotenv()

API_KEY = os.getenv("USDA_API_KEY")

# Compute project root dynamically, regardless of cwd
PROJECT_ROOT       = Path(__file__).resolve().parent.parent

#Hard-coded settings
CONFIDENCE_THRESHOLD = 0.25

# Data & model paths (relative to project root)
RAW_IMAGES_DIR       = PROJECT_ROOT / "data" / "images"
TRAIN_DIR            = PROJECT_ROOT / "data" / "train"
VAL_DIR              = PROJECT_ROOT / "data" / "validation"
TEST_DIR             = PROJECT_ROOT / "data" / "test"

MODEL_PATH           = PROJECT_ROOT / "models" / "best_model_101class.hdf5"
NUTRITION_CSV_PATH   = PROJECT_ROOT / "data" / "food_nutrition.csv"
TEMP_NUTRITION_JSON  = PROJECT_ROOT / "data" / "temp_nutrition.json"

# Training history log
HISTORY_LOG_PATH     = PROJECT_ROOT / "models" / "history_101class.log"

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
