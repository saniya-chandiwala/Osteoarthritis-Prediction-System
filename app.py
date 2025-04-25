import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import transforms
from timm import create_model
import os
import pandas as pd

# === MODEL SETUP ===

MODEL_FOLDER = "C:/Users/Asus/Downloads/project/model"

model_files = {
    "DINO_OG": "DINO_OG.pth",
    "DINO_Distributed": "DINO_dist.pth",
    "InceptionV3_OG": "IV3_OG.keras",
    "InceptionV3_Distributed": "IV3_dist.keras",
    "CoAtNet_OG": "CoAtNet_OG.keras",
    "CoAtNet_Distributed": "CoAtNet_dist.keras",
    "EfficientNet_OG": "effi_OG.keras",
    "EfficientNet_Distributed": "effi_dist.keras"
}

# === DEFINE DINO MODEL CLASS ===

class OsteoarthritisClassifier(nn.Module):
    def __init__(self, backbone):
        super(OsteoarthritisClassifier, self).__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, 1)

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        return torch.sigmoid(self.fc(features))

def load_dino_model(weights_path):
    backbone = create_model("vit_base_patch16_224_dino", pretrained=True)
    backbone.head = nn.Identity()
    model = OsteoarthritisClassifier(backbone)
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# === LOAD MODELS ===

keras_models = {}
input_sizes = {}

for name, file in model_files.items():
    if file.endswith(".keras"):
        full_path = os.path.join(MODEL_FOLDER, file)
        keras_models[name] = tf.keras.models.load_model(full_path)
        if "InceptionV3" in name:
            input_sizes[name] = (128, 128)
        else:
            input_sizes[name] = (224, 224)

dino_models = {}
for name, file in model_files.items():
    if file.endswith(".pth"):
        full_path = os.path.join(MODEL_FOLDER, file)
        dino_models[name] = load_dino_model(full_path)

# === IMAGE PREPROCESSING ===

def preprocess_for_keras(img, size):
    img = img.resize(size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_for_torch(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(img).unsqueeze(0)

# === PREDICTION FUNCTION ===

def get_predictions(image):
    results = {}

    for name, model in keras_models.items():
        size = input_sizes.get(name, (224, 224))
        input_img = preprocess_for_keras(image, size)
        pred = model.predict(input_img, verbose=0)[0][0] * 100
        results[name] = round(pred, 2)

    for name, model in dino_models.items():
        input_tensor = preprocess_for_torch(image)
        with torch.no_grad():
            pred = model(input_tensor).item() * 100
            results[name] = round(pred, 2)

    return results

# === FORMAT PREDICTIONS INTO A TABLE ===

def format_prediction_table(predictions):
    rows = []
    base_models = ["DINO", "InceptionV3", "CoAtNet", "EfficientNet"]

    for model in base_models:
        original = predictions.get(f"{model}_OG", "N/A")
        distributed = predictions.get(f"{model}_Distributed", "N/A")
        rows.append({
            "Model": model,
            "Original Dataset": f"{original}%" if isinstance(original, (int, float)) else original,
            "Rearranged Dataset": f"{distributed}%" if isinstance(distributed, (int, float)) else distributed
        })

    return pd.DataFrame(rows)

# === STREAMLIT UI ===

st.title("Osteoarthritis Prediction System")
st.write("Upload an X-ray image to get predictions.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            predictions = get_predictions(image)

        st.write("### Predictions from all models (in %):")
        st.table(format_prediction_table(predictions))
