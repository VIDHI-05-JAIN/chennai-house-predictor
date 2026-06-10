import joblib
model = joblib.load("/Users/admin/Desktop/ML_Project/chennai-house-predictor/model/house_price_pipe.pkl")

try:
    print(model.feature_names_in_)
except Exception as e:
    print("No attribute 'feature_names_in_' found.")
    print("Type of model:", type(model))
