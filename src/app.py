import streamlit as st
import tensorflow as tf
import pandas as pd

from utils import load_and_prep, get_classes
from food_descriptions import get_food_descriptions
from get_nutrition_data import process_nutrition_data
import config
import ui

# Page config
st.set_page_config(page_title="Food Recognition", page_icon="🍽️", layout="wide")

# Sidebar
st.sidebar.header("Overview")
st.sidebar.markdown(
    f"- **Confidence threshold:** **`{config.CONFIDENCE_THRESHOLD:.2f}`**  \n"
    "- **Accuracy:** 	**`83.6%`**  \n"
    "- **Model:** **`InceptionV3 on Food-101`**  \n"
    "- **Nutrition data:** **`USDA API`**"
)

# Load nutrition data
with st.spinner("Preparing nutrition data..."):
    process_nutrition_data(
        api_key=config.API_KEY,
        output_path=config.NUTRITION_CSV_PATH,
        temp_json_path=config.TEMP_NUTRITION_JSON
    )

# Load model & data into session
if "model" not in st.session_state:
    with st.spinner("Loading model..."):
        st.session_state.model = tf.keras.models.load_model(config.MODEL_PATH)
if "nutrition_df" not in st.session_state:
    st.session_state.nutrition_df = pd.read_csv(config.NUTRITION_CSV_PATH)
if "class_names" not in st.session_state:
    st.session_state.class_names = get_classes()

model        = st.session_state.model
nutrition_df = st.session_state.nutrition_df
class_names  = st.session_state.class_names

# Main Title
st.markdown("<h1 style='color:#386641;'>Food Image Classification & Nutritional Analysis</h1>", unsafe_allow_html=True)

# Image uploader
uploaded = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.info("Please upload an image to continue.")
    st.stop()

# Prediction
raw   = uploaded.read()
img   = load_and_prep(raw)
probs = model.predict(tf.expand_dims(img, axis=0))[0]

# Build top-5 DataFrame
results = (
    pd.DataFrame({"label": class_names, "confidence": probs})
      .sort_values("confidence", ascending=False)
      .head(5)
)
results["confidence (%)"] = (results["confidence"] * 100).round(2)

best_label = results.iloc[0]["label"]
best_conf  = results.iloc[0]["confidence"]

# Layout: image+pred vs top-5
c1, c2 = st.columns([1,1])

with c1:
    ui.display_image_and_prediction(uploaded, best_label, best_conf)
    desc = get_food_descriptions().get(best_label, "No description available.")
    ui.show_description(desc)

with c2:
    st.markdown("<h2 style='color:#6a994e;'>Top-5 Predictions</h2>", unsafe_allow_html=True)
    ui.plot_top5_confidence(results[["label","confidence (%)"]])

# Threshold check
if best_conf < config.CONFIDENCE_THRESHOLD:
    st.warning("Confidence is below threshold; result may be unreliable.")
    st.stop()

# Nutrition & Recommendations
st.markdown("---")
st.markdown("<h2 style='color:#6a994e;'>Nutrition & Recommendations</h2>", unsafe_allow_html=True)

info   = nutrition_df.query("label == @best_label").drop(columns=["label"])
protein, fat, carb = info.iloc[0][["protein","fat","carbohydrates"]]

colA, colB = st.columns(2)
with colA:
    ui.show_nutritional_facts(info)
    recs = ui.generate_recommendation(info.iloc[0])
    ui.show_recommendations(recs)
with colB:
    st.markdown("<h4 style='color:#a7c957;'>Macronutrient Distribution</h4>", unsafe_allow_html=True)
    ui.plot_macronutrient_pie({
        "Protein": protein,
        "Fat": fat,
        "Carbohydrates": carb
    })
