from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
model = joblib.load("models/81_R01235mlp_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Disease classes
class_names = {
    4: "Gall Midge",
    6: "Powdery Mildew",
    7: "Sooty Mould",
    8: "rot",
    9: "burn"
}

def rgb_to_hue(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    r, g, b = cv2.split(image)
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g)**2 + (r - b)*(g - b)) + 1e-6
    theta = np.arccos(num / den)
    h = np.where(b > g, 2 * np.pi - theta, theta) / (2 * np.pi)
    return h

def mask_green_pixels(hue_image):
    green_mask = (hue_image > 0.25) & (hue_image < 0.45)
    return np.where(green_mask, 0, hue_image)

def extract_features(image):
    hue = rgb_to_hue(image)
    masked = mask_green_pixels(hue)
    binary = np.where(masked > 0, 1, 0).astype(np.uint8)

    infected_ratio = binary.sum() / binary.size
    infected_hues = hue[binary == 1]
    mean_hue = infected_hues.mean() if infected_hues.size > 0 else 0
    std_hue = infected_hues.std() if infected_hues.size > 0 else 0

    edges = cv2.Canny((binary * 255).astype(np.uint8), 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours)
    num_contours = len(contours)

    return [infected_ratio, perimeter, mean_hue, std_hue, num_contours]

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    photo = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    features = extract_features(photo)
    if features is not None:
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        predicted_class = class_names.get(prediction, "Unknown")
        return JSONResponse({"class": predicted_class})
    return JSONResponse({"error": "Failed to process image"}, status_code=400)