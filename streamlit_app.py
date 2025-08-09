import streamlit as st
import pandas as pd
import joblib

# Load the model and encoder
@st.cache_resource
def load_model():
    pipeline = joblib.load('final_menu_profitability_pipeline.pkl')
    le = joblib.load('profitability_label_encoder.pkl')
    return pipeline, le

model, label_encoder = load_model()

# App title
st.title("🍽️ Menu Profitability Predictor")

# Input form
with st.form("prediction_form"):
    st.header("Enter Menu Details")
    
    restaurant = st.selectbox("Restaurant ID", ["R001", "R002", "R003"])
    category = st.selectbox("Menu Category", ["Appetizers", "Beverages", "Desserts", "Main Course"])
    price = st.number_input("Price ($)", min_value=0.0, step=0.01)
    ingredients = st.number_input("Number of Ingredients", min_value=1, step=1)
    name_length = st.number_input("Menu Item Name Length", min_value=1, step=1)
    
    submitted = st.form_submit_button("Predict Profitability")

# Make prediction
if submitted:
    input_data = pd.DataFrame({
        'RestaurantID': [restaurant],
        'MenuCategory': [category],
        'Price': [price],
        'IngredientCount': [ingredients],
        'MenuItemLength': [name_length]
    })
    
    prediction = model.predict(input_data)
    result = label_encoder.inverse_transform(prediction)[0]
    
    st.success(f"Predicted Profitability: **{result}**")
    
    # Show confidence scores
    proba = model.predict_proba(input_data)[0]
    st.subheader("Confidence Scores")
    for i, class_name in enumerate(label_encoder.classes_):
        st.write(f"{class_name}: {proba[i]*100:.1f}%")
    
    # Simple bar chart
    st.bar_chart(pd.DataFrame({
        'Class': label_encoder.classes_,
        'Confidence': proba
    }).set_index('Class'))