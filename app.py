from flask import Flask, request, Response
from twilio.twiml.messaging_response import MessagingResponse
import pandas as pd
import ee
import joblib
import re
import os
import traceback
from google.oauth2 import service_account
import datetime
import random   

MODEL_PATH = "xgb_model.pkl"
FEATURE_COLUMNS = ["Rainfall (mm)", "Depth_to_Water_Level (m)", "Soil_Type", "Rock_Type"]
SERVICE_ACCOUNT_KEY = "service-account-key.json"

DATASET_CONFIG = {
    "dem": "USGS/SRTMGL1_003",
    "rainfall_collection": "UCSB-CHG/CHIRPS/DAILY",
    "soil": "OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02",
    "geology": "CSP/ERGo/1_0/Global/ALOS_topoDiversity"
}

SOIL_MAP = {0: "Clay", 1: "Silt", 2: "Loamy", 3: "Sandy Loam", 4: "Sandy"}
ROCK_MAP = {0: "Granite", 1: "Basalt", 2: "Gneiss", 3: "Sandstone", 4: "Limestone"}

app = Flask(__name__)

def initialize_gee():
    try:
        if not os.path.exists(SERVICE_ACCOUNT_KEY):
            print(f"❌ Service account key not found at {SERVICE_ACCOUNT_KEY}")
            return False
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_KEY,
            scopes=["https://www.googleapis.com/auth/earthengine"]
        )
        ee.Initialize(credentials=credentials)
        print("✅ GEE Initialized with Service Account")
        return True
    except Exception as e:
        print(f"❌ GEE Initialization failed: {e}")
        traceback.print_exc()
        return False

gee_available = initialize_gee()

try:
    xgb_model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    xgb_model = None
    print(f"❌ Failed to load model: {e}")
    traceback.print_exc()

def parse_coordinates(text):
    nums = re.findall(r"-?\d+\.\d+|-?\d+", text)
    if len(nums) < 2:
        return None
    a, b = float(nums[0]), float(nums[1])
    if -90 <= a <= 90 and -180 <= b <= 180:
        return (a, b)
    if -90 <= b <= 90 and -180 <= a <= 180:
        return (b, a)
    return (a, b)

SOIL_BUFFER_MAP = {
    "Clay": 15,
    "Sandy": 10,
    "Loamy": 15,
    "Rocky": 18,
    "Gravelly": 19,
    "Silt": 20,
    "Unknown": 20
}

def safe_sample(image, point, scale, band, default=0):
    try:
        sample = image.sample(point, scale).first()
        if sample:
            val = sample.get(band).getInfo()
            return val if val is not None else default
        else:
            return default
    except Exception:
        return default

def get_rainfall_mean(point, days=30):
    """Fetch average rainfall over last N days (mocked as random for testing)."""
    
    rainfall_val = round(random.uniform(1.0, 10.0), 2)
    return rainfall_val

def get_features(lat, lon):
    point = ee.Geometry.Point([lon, lat])

    rain_val = get_rainfall_mean(point, days=30)

    slope_val = safe_sample(ee.Terrain.slope(ee.Image(DATASET_CONFIG["dem"])), point, 30, "slope", default=0)

    soil_val = safe_sample(ee.Image(DATASET_CONFIG["soil"]), point, 250, "b0", default=0)
    soil_str = SOIL_MAP.get(int(round(float(soil_val))), "Unknown")

    rock_val = safe_sample(ee.Image(DATASET_CONFIG["geology"]), point, 250, "constant", default=0)
    rock_str = ROCK_MAP.get(int(round(float(rock_val))), "Unknown")

    features_dict = {
        "Rainfall (mm)": rain_val,
        "Depth_to_Water_Level (m)": slope_val,
        "Soil_Type": soil_str,
        "Rock_Type": rock_str
    }

    soil_buffer = SOIL_BUFFER_MAP.get(soil_str, 20)

    if xgb_model is not None:
        X_pred = pd.DataFrame([features_dict])[FEATURE_COLUMNS]
        predicted_depth = float(xgb_model.predict(X_pred)[0] + soil_buffer)
        features_dict["Predicted_Bore_Depth (m)"] = predicted_depth
    else:
        features_dict["Predicted_Bore_Depth (m)"] = None

    print("📊 Feature summary:", features_dict)
    return features_dict

@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    resp = MessagingResponse()
    msg = resp.message()

    lat = request.form.get("Latitude")
    lon = request.form.get("Longitude")

    if lat and lon:
        try:
            lat, lon = float(lat), float(lon)
        except:
            lat = lon = None
    else:
        body = request.form.get("Body", "")
        parsed = parse_coordinates(body)
        if parsed:
            lat, lon = parsed
        else:
            lat = lon = None

    if lat is None or lon is None:
        reply = "❌ Could not parse coordinates. Send as text (lat,lon) or share location."
        msg.body(reply)
        return Response(str(resp), mimetype="application/xml")

    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        reply = f"❌ Coordinates out of range: ({lat}, {lon})"
        msg.body(reply)
        return Response(str(resp), mimetype="application/xml")

    features = get_features(lat, lon)
    if features and features.get("Predicted_Bore_Depth (m)") is not None:
        reply = (
            f"📍 Location: ({lat:.6f}, {lon:.6f})\n"
            f"🌧 Rainfall (30-day mean): {features['Rainfall (mm)']:.2f} mm\n"
            f"📈 Slope_Index: {features['Depth_to_Water_Level (m)']:.2f} m\n"
            f"🪨 Rock: {features['Rock_Type']}\n"
            f"💧 Predicted Bore Depth: {features['Predicted_Bore_Depth (m)']:.2f} m"
        )
    else:
        reply = "❌ Could not fetch features or model not loaded."

    msg.body(reply)
    print("📤 Replying:", reply)
    return Response(str(resp), mimetype="application/xml")

@app.route("/health", methods=["GET"])
def health_check():
    return {
        "status": "healthy",
        "model_loaded": xgb_model is not None,
        "gee_available": gee_available
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
