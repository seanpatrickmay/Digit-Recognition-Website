import numpy as np
from PIL import Image
import sys

def process_png(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((28,28))
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=-1)

    return image_array

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 process-png.py 'image_path' 'output_path'")
        sys.exit(1)
        
    image_path = sys.argv[1]
    output_path = sys.argv[2]

    try:
        processed_png = process_png(image_path)
        np.save(output_path, processed_png)

    except Exception as e:
        print(f"An error has occured: {e}")
        sys.exit(1)
