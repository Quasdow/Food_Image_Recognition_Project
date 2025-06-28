import streamlit as st
import plotly.express as px
import pandas as pd

# Color palettes
BAR_COLORS = ["#6f2dbd","#a663cc","#b298dc","#b8d0eb","#b9faf8"]
PIE_COLORS = {"Fat":"#26547c","Protein":"#ef476f","Carbohydrates":"#ffd166"}

# Choose text color based on confidence
def _conf_color(conf: float) -> str:
    if conf < 0.25:
        return "#e76f51"
    elif conf < 0.50:
        return "#f4a261"
    else:
        return "#2a9d8f"

def display_image_and_prediction(uploaded_file, label: str, confidence: float):
    """Show image and colored prediction text."""
    st.image(uploaded_file, use_column_width=True)
    color = _conf_color(confidence)
    percent = confidence * 100

    st.markdown(
    f"<h2 style='color:{color};'>{label} ({percent:.2f}%)</h2>",
    unsafe_allow_html=True
)

def plot_top5_confidence(df: pd.DataFrame):
    fig = px.bar(
        df,
        x="confidence (%)",
        y="label",
        orientation="h",
        color="label",
        color_discrete_sequence=BAR_COLORS,
        template="plotly_white"
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        margin=dict(l=80, r=20, t=10, b=10),
        xaxis_title="Confidence (%)"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_nutritional_facts(df: pd.DataFrame):
    st.markdown("<h4 style='color:#a7c957;'>Nutritional Facts (per 100 g)</h4>", unsafe_allow_html=True)
    fmt_df = df.copy()
    for col in ["calories", "protein", "fat", "carbohydrates", "vitamins", "calcium"]:
        if col in fmt_df:
            fmt_df[col] = fmt_df[col].map(lambda x: f"{x:.1f}")
    st.dataframe(fmt_df, use_container_width=True)

def plot_macronutrient_pie(macros: dict):
    df = pd.DataFrame({"Nutrient":list(macros.keys()),"Value":list(macros.values())})
    fig = px.pie(
        df, names="Nutrient", values="Value", hole=0.4,
        color="Nutrient", color_discrete_map=PIE_COLORS,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_description(desc: str):
    """Expandable section for the food description."""
    with st.expander("Description"):
        st.write(desc)

def generate_recommendation(n: pd.Series) -> list[str]:
    """
    Generate a diverse set of dietary suggestions based on nutritional thresholds.
    """
    recs = []
    
    # Calories-based suggestions
    if n.calories > 600:
        recs += [
            "High in energy—pair with fiber-rich vegetables like broccoli or spinach.",
            "Portion control is key; try using smaller plates to manage intake."
        ]
    elif n.calories < 200:
        recs += [
            "Low in calories—boost with healthy fats like avocado or olive oil.",
            "Add a side of whole grains or legumes for sustained energy."
        ]
    
    # Protein-based suggestions
    if n.protein < 5:
        recs += [
            "Protein is low—consider adding grilled chicken, tofu, or lentils.",
            "A side of Greek yogurt or a handful of almonds can help meet protein needs."
        ]
    elif n.protein > 25:
        recs += [
            "Excellent source of protein—ideal for muscle recovery after exercise.",
            "Pair with complex carbs like quinoa or sweet potatoes for balanced nutrition."
        ]
    
    # Fat-based suggestions
    if n.fat > 25:
        recs += [
            "High in fat—limit additional fats today; opt for grilled or steamed sides.",
            "Stay hydrated with water or herbal tea to aid digestion."
        ]
    elif n.fat < 5:
        recs += [
            "Low in fat—incorporate healthy fats like nuts, seeds, or a drizzle of olive oil.",
            "Consider adding avocado slices for a creamy texture and heart-healthy fats."
        ]
    
    # Carbohydrates-based suggestions
    if n.carbohydrates > 60:
        recs += [
            "Carbohydrates are high—balance with protein-rich foods like eggs or fish.",
            "Reduce starchy sides; opt for non-starchy vegetables instead."
        ]
    elif n.carbohydrates < 15:
        recs += [
            "Low in carbohydrates—add whole grains like brown rice or oats for energy.",
            "Perfect for low-carb diets; ensure you're meeting energy needs with other nutrients."
        ]
    
    # Micronutrient-based suggestions
    if n.vitamins < 0.5:
        recs.append("Vitamin content is low—consider adding citrus fruits or bell peppers for vitamin C.")
    if n.calcium < 100:
        recs.append("Calcium is low—incorporate dairy, fortified plant milks, or leafy greens.")
    
    # Additional combination-based suggestions
    if n.calories > 600 and n.fat > 25:
        recs.append("This dish is high in both calories and fat—opt for a lighter, vegetable-based side.")
    if n.protein > 25 and n.carbohydrates < 15:
        recs.append("High protein and low carb—great for muscle building or weight management.")
    
    # Fallback suggestion
    if not recs:
        recs.append("Overall, this dish has a balanced nutritional profile—enjoy in moderation.")

    return recs

def show_recommendations(recs: list[str]):
    st.markdown("<h4 style='color:#a7c957;'>Dietary Recommendations</h4>", unsafe_allow_html=True)
    for i, r in enumerate(recs, start=1):
        st.markdown(f"{i}. {r}")
