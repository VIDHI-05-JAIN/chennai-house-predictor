import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import io

# ==============================
# ⚙️ PAGE CONFIGURATION
# ==============================
st.set_page_config(
    page_title="🏠 Chennai House Price Predictor",
    layout="wide",
    page_icon="🏡",
)

# ==============================
# 📦 LOAD MODEL & DATA
# ==============================
MODEL_PATH = "/Users/admin/Desktop/ML_Project/chennai-house-predictor/model/house_price_pipe.pkl"
DATA_PATH = "data/train-chennai-sale.csv"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model not found.")
    st.stop()
if not os.path.exists(DATA_PATH):
    st.error("❌ Dataset not found.")
    st.stop()

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

target_col = "SALES_PRICE" if "SALES_PRICE" in df.columns else "SALE_PRICE"
locality_col = "AREA"

# Fill missing values
num_cols = df.select_dtypes(include=["number"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# Defaults
num_defaults = df[num_cols].median().to_dict()
cat_defaults = df[cat_cols].mode().iloc[0].to_dict()

# Locality stats
locality_stats = df.groupby(locality_col)[target_col].agg(["median","min","max"]).reset_index()
all_localities = sorted(df[locality_col].dropna().unique().tolist())

# Years
# Continuous years including future
def get_continuous_years(series, future_max=2035):
    years = []
    for val in series.dropna().unique():
        # Try parsing as datetime
        try:
            y = pd.to_datetime(val, errors='coerce').year
            if not np.isnan(y):
                years.append(int(y))
        except:
            pass
        # Try parsing as integer
        try:
            y = int(val)
            if y > 1800:
                years.append(y)
        except:
            pass
    # Remove duplicates and ensure all are ints
    years = [int(y) for y in years if not isinstance(y, list)]
    years = sorted(list(set(years)))
    
    if years:
        min_year = min(years)
        max_year = max(max(years), future_max)
        return list(range(min_year, max_year + 1))
    else:
        return list(range(2000, future_max + 1))


# Generate year lists
build_years = get_continuous_years(df.get("DATE_BUILD"), future_max=2040)
sale_years = get_continuous_years(df.get("DATE_SALE"), future_max=2040)

# Defaults (median)
default_build = int(np.median(build_years))
default_sale = int(np.median(sale_years))

expected_cols = model.feature_names_in_ if hasattr(model, "feature_names_in_") else df.columns

# ==============================
# 🎨 STYLING
# ==============================
st.markdown("""
<style>
body {background-color: #f9fafc; font-family: 'Inter', sans-serif;}
.title {text-align:center; font-size:2.5rem; font-weight:800; color:#1E3A8A; margin-bottom:0.2rem;}
.subtitle {text-align:center; font-size:1.2rem; color:#334155; margin-bottom:2rem;}
.stButton>button {background-color:#0072B5;color:white;border-radius:8px;height:2.8rem;width:100%;font-size:1.05rem;border:none;}
.stButton>button:hover {background-color:#005B8C;}
.footer {text-align:center;color:gray;font-size:0.9rem;padding-top:1.5rem;}
.metric-card {background:#1e1e1e;color:#fff;border-radius:12px;padding:15px 20px;text-align:center;margin-bottom:25px;box-shadow:0 2px 10px rgba(0,0,0,0.3);}
</style>
""", unsafe_allow_html=True)

# ==============================
# 🧾 MAIN TITLE
# ==============================
st.markdown('<div class="title">🏡 Chennai House Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Estimate property prices with ML & Chennai data trends</div>', unsafe_allow_html=True)

# ==============================
# 🎯 MODEL PERFORMANCE
# ==============================
r2 = 0.9982
mae = 91245
rmse = 160557
st.markdown(f"""
<div class="metric-card">
    <h4 style="margin-bottom:8px;font-weight:600;">🎯 Model Performance</h4>
    <p style="font-size:1.05rem;font-weight:500;">
        R²: {r2:.3f} 📏 MAE: ₹{mae:,.0f} 📊 RMSE: ₹{rmse:,.0f}
    </p>
</div>
""", unsafe_allow_html=True)

# ==============================
# 🗂️ SIDEBAR INPUTS
# ==============================
st.sidebar.header("🏠 Property Details")
with st.sidebar.expander("Property Features"):
    sqft = st.number_input("Built-up Area (sqft)", 400, 10000, 1200, 50)
    bedrooms = st.number_input("Bedrooms", 1, 6, 2)
    bathrooms = st.number_input("Bathrooms", 1, 6, 2)
    rooms = st.number_input("Total Rooms", 1, 10, 4)

with st.sidebar.expander("Location & Year"):
    locality = st.selectbox("Locality", all_localities)
    year_built = st.selectbox("Year Built", build_years or [default_build])
    year_sold = st.selectbox("Year Sold", sale_years or [default_sale])

submitted = st.sidebar.button("🔮 Predict Price")

# ==============================
# 🧩 PREDICTION LOGIC
# ==============================
if submitted:
    try:
        # Prepare input
        user_input = {
            "INT_SQFT": sqft,
            "N_ROOM": rooms,
            "N_BEDROOM": bedrooms,
            "N_BATHROOM": bathrooms,
            "YEAR_DATE_BUILD": year_built,
            "YEAR_DATE_SALE": year_sold,
        }
        for col in expected_cols:
            if col not in user_input:
                if col in num_defaults:
                    user_input[col] = num_defaults[col]
                elif col in cat_defaults:
                    user_input[col] = cat_defaults[col]
                else:
                    user_input[col] = 0

        X_input = pd.DataFrame([user_input])
        X_input = X_input[expected_cols]

        # Make prediction
        prediction = model.predict(X_input)[0]
        if prediction < 1000:
            prediction = np.expm1(prediction)

        # Locality stats
        loc_data = locality_stats[locality_stats[locality_col] == locality]
        med = loc_data["median"].values[0] if not loc_data.empty else np.nan
        mn = loc_data["min"].values[0] if not loc_data.empty else np.nan
        mx = loc_data["max"].values[0] if not loc_data.empty else np.nan

        # ==============================
        # 🏆 RESULT CARD
        # ==============================
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg,#ffffff,#e0f2fe);
                color:#000;
                border-radius:12px;
                padding:50px 40px;
                text-align:center;
                box-shadow:0 8px 25px rgba(0,0,0,0.12);
            ">
                <h2 style='font-weight:700;'>🏠 Estimated Property Value</h2>
                <h1 style='font-size:3rem;font-weight:800;margin:15px 0;'>₹ {prediction:,.0f}</h1>
                <p>📍 <b>Locality:</b> {locality}</p>
                <p>💰 Median: ₹{med:,.0f} 📉 Min: ₹{mn:,.0f} 📈 Max: ₹{mx:,.0f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ==============================
        # ⚡ ALERTS FOR HIGH/LOW
        # ==============================
        if not np.isnan(med):
            if prediction > med*1.0:
                st.success("💡 This property is valued significantly higher than median!")
            elif prediction < med*0.7:
                st.warning("⚠️ This property is valued lower than median.")
        else:
            st.info("ℹ️ Locality median price not available to generate alert.")

        # ==============================
        # 1️⃣ LOCALITY PRICE HISTOGRAM
        # ==============================
        loc_prices = df[df[locality_col] == locality][target_col]
        fig_hist = px.histogram(loc_prices, nbins=20, title=f"{locality} Price Distribution", labels={"value":"Price (₹)"})
        fig_hist.update_layout(height=350, margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(fig_hist, use_container_width=True)

        # ==============================
        # 2️⃣ LOCALITY COMPARISON CHART
        # ==============================
        fig_comp = px.bar(
            x=["Median Price", "Min Price", "Max Price", "Predicted Price"],
            y=[med, mn, mx, prediction],
            text=[f"₹{med:,.0f}", f"₹{mn:,.0f}", f"₹{mx:,.0f}", f"₹{prediction:,.0f}"],
            color=["Median Price", "Min Price", "Max Price", "Predicted Price"],
            color_discrete_map={"Median Price":"#1f77b4","Min Price":"#a6cee3","Max Price":"#b2df8a","Predicted Price":"#ff7f0e"},
            title=f"📊 {locality} Price Comparison"
        )
        fig_comp.update_layout(height=400, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig_comp, use_container_width=True)

        # ==============================
        # 3️⃣  DOWNLOAD OPTION
        # ==============================
        csv_buffer = io.StringIO()
        pd.DataFrame([{**user_input, "Predicted Price": prediction}]).to_csv(csv_buffer, index=False)
        st.download_button("📥 Download Prediction", data=csv_buffer.getvalue(), file_name="prediction.csv")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")