"""

import os
import pandas as pd

train_images_folder = "train"
classes_csv_file = "train.csv"

df = pd.read_csv(classes_csv_file)

dataset = []

classes_link = [
"Buildings",
"Forests",
"Mountains",
"Glacier",
"Sea",
"Street"]

for _, row in df.iterrows():
    image_path = train_images_folder + "/" + row['image_name']
    label = classes_link[int(row['label'])]
    dataset.append((image_path, label))


print(dataset[:5])


import os
import random
from PIL import Image, ImageDraw, ImageFont

def overlay_text_on_image(image_path, label, output_folder="augmented_images"):
    
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    font_size = random.randint(20, 40)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    text_width, text_height = draw.textsize(label, font=font)


    max_x = max(1, width - text_width)
    max_y = max(1, height - text_height)
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    # Random text color
    text_color = tuple(random.randint(155, 255) for _ in range(3))

    # Draw text
    draw.text((x, y), label, fill=text_color, font=font)

    # Save image
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    image.save(output_path)

    # Bounding box: (x1, y1, x2, y2)
    bbox = (x, y, x + text_width, y + text_height)

    return output_path, bbox

for image_path, label in dataset:
    new_image_path, bbox = overlay_text_on_image(image_path, label)
    print(f"{bbox}")

"""

from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data="dataset.yaml",  # Path to dataset configuration file
    epochs=1,  # Number of training epochs
    imgsz=640,  # Image size for training
    device="cuda",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)

# Evaluate the model's performance on the validation set
metrics = model.val()

# Export the model to ONNX format for deployment
path = model.export(format="onnx")  # Returns the path to the exported model


