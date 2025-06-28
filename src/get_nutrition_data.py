import os
import time
import json
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils import get_classes
import config

def get_nutrition_from_api(food_name: str, retries: int = 3, backoff_factor: float = 1.0) -> dict:
    """
    Fetch macronutrient data for a single food_name from USDA API.
    Tries variants of food_name; returns zeros if all fail.
    """
    api_key = config.API_KEY
    if not api_key:
        raise RuntimeError("USDA_API_KEY is missing; set in .env")

    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retry_strategy))

    variants = [
        food_name.replace("_", " "),
        food_name.replace("_", "-"),
        food_name
    ]
    for variant in variants:
        response = session.get(
            "https://api.nal.usda.gov/fdc/v1/foods/search",
            params={"query": variant, "pageSize": 1, "api_key": api_key}
        )
        if response.ok:
            data = response.json()
            if foods := data.get("foods"):
                nutrients = {
                    nut["nutrientName"].lower(): nut["value"]
                    for nut in foods[0].get("foodNutrients", [])
                }
                return {
                    "label": food_name,
                    "protein": nutrients.get("protein", 0.0),
                    "calcium": nutrients.get("calcium", 0.0),
                    "fat": nutrients.get("total lipid (fat)", 0.0),
                    "carbohydrates": nutrients.get("carbohydrate, by difference", 0.0),
                    "vitamins": nutrients.get("vitamin c, total ascorbic acid", 0.0),
                    "calories": nutrients.get("energy", 0)
                }
        time.sleep(1)

    print(f"[Warning] No data found for {food_name}")
    return {
        "label": food_name,
        "protein": 0.0,
        "calcium": 0.0,
        "fat": 0.0,
        "carbohydrates": 0.0,
        "vitamins": 0.0,
        "calories": 0
    }

def process_nutrition_data(api_key=None, output_path=None, temp_json_path=None) -> None:
    """
    Build or load the nutrition CSV. If CSV already exists, skip.
    Otherwise fetch each class via USDA API and save to CSV.
    """
    output_path    = output_path    or config.NUTRITION_CSV_PATH
    temp_json_path = temp_json_path or config.TEMP_NUTRITION_JSON

    if os.path.exists(output_path):
        print(f"[Info] {output_path} already exists; skipping processing.")
        return

    classes = get_classes()
    results = []
    for idx, label in enumerate(classes, start=1):
        print(f"[Fetch] {idx}/{len(classes)}: {label}")
        info = get_nutrition_from_api(label)
        results.append(info)
        # Save interim JSON in case of interruption
        with open(temp_json_path, "w") as f:
            json.dump(results, f, indent=2)

    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"[Success] Nutrition data saved to {output_path}")
    if os.path.exists(temp_json_path):
        os.remove(temp_json_path)
