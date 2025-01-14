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

MODEL_PATH = "test_model.pth"

# Set device
device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
)
print(f"Using {device} as device")

model = ConvolutionalNeuralNetwork().to(device)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

def process_image(image_path):
    image = Image.open(image_path).convert("L")
    image = image.resize((28,28))
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=(0, -1))
    return torch.from_numpy(image_array).permute(0, 3, 1, 2)



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 predict-image.py 'image_path'")
        sys.exit(1)

    image_path = sys.argv[1]

    input_tensor = process_image(image_path).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        print(f"Predicted class: {predicted_class}")


