import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import ssl
from ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from PIL import Image
from flask import Flask, request, jsonify
import io
import base64
from flask_cors import CORS

MODEL_PATH = "convolutional_model.pth"

app = Flask(__name__)
CORS(app)

# Set device
device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
)

# Set up model, load weights, put to eval mode
model = ConvolutionalNeuralNetwork().to(device)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

def process_image_file(image):
    print("Trying process_image_file method")
    image = image.convert("L")
    image = image.resize((28,28))
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=(0, -1))
    return torch.from_numpy(image_array).permute(0, 3, 1, 2)

def process_image_data(data):
    print("Trying process_image_data method")
    image_data = base64.b64decode(data.split(",")[1])
    return Image.open(io.BytesIO(image_data))

def model_predict(input_tensor):
    print("Trying model_predict method")
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output).item()
        print(f"Predicted class found: {predicted_class}")
        return predicted_class

@app.route("/predict", methods=["POST"])
def predict():
    print("Trying predict method")
    try:
        data = request.json.get("image")
        if not data:
            return jsonify({"error": "No image provided"}), 400
        image_data = process_image_data(data)
        image_tensor = process_image_file(image_data)
        predicted_digit = model_predict(image_tensor)
        print(f"Model found predicted digit to be: {predicted_digit}")
        return jsonify({"prediction": predicted_digit})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate", methods=["POST"])
def generate():
    return
# Method to generate 
    
if __name__ == "__main__":
    app.run(debug=True)
