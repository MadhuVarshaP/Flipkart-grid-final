import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json
import pandas as pd
from datetime import datetime

# Load the saved model
model_path = 'my_model.keras'
try:
    model = load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model. Error: {e}")
    exit()

# Load the class indices (the mapping from class labels to indices)
class_indices_path = 'class_indices.json'
try:
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
        print("Class indices loaded successfully!")
except Exception as e:
    print(f"Failed to load class indices. Error: {e}")
    exit()

# Reverse the mapping to go from index to class label
label_map = {v: k for k, v in class_indices.items()}

# Function to preprocess a single image
def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))  # Resize image to input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize pixel values
        return img_array
    except Exception as e:
        print(f"Error in preprocessing image {img_path}. Error: {e}")
        return None

# Function to predict and update results in an Excel file
def bulk_test_model(image_folder, excel_file):
    # Check if the folder exists
    if not os.path.exists(image_folder):
        print(f"Folder does not exist: {image_folder}")
        return

    # List all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    if not image_files:
        print(f"No image files found in the folder: {image_folder}")
        return

    # Check if the Excel file already exists
    if os.path.exists(excel_file):
        df = pd.read_excel(excel_file)
    else:
        df = pd.DataFrame(columns=['S. No.', 'Product Name', 'Time', 'Count'])

    # Process each image file
    for img_name in image_files:
        img_path = os.path.join(image_folder, img_name)
        print(f"Processing image: {img_name}")

        # Preprocess the image
        img_array = preprocess_image(img_path)
        if img_array is None:
            continue

        # Predict the brand
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        predicted_label = label_map[predicted_class]

        # Get current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Check if the product is already in the Excel file
        if predicted_label in df['Product Name'].values:
            df.loc[df['Product Name'] == predicted_label, 'Count'] += 1
            df.loc[df['Product Name'] == predicted_label, 'Time'] = current_time
        else:
            new_row = pd.DataFrame({
                'S. No.': [len(df) + 1],
                'Product Name': [predicted_label],
                'Time': [current_time],
                'Count': [1]
            })
            df = pd.concat([df, new_row], ignore_index=True)

        # Display the prediction result
        print(f"Image: {img_name}")
        print(f"Predicted Brand: {predicted_label}")
        print(f"Confidence: {np.max(predictions[0]) * 100:.2f}%")
        print("-" * 50)

    # Save the updated dataframe to Excel
    df.to_excel(excel_file, index=False)
    print(f"Results updated in {excel_file}")

# Specify the folder containing the test images and the Excel file to update
test_image_folder = r'C:\Users\uvara\OneDrive\Desktop\lets begin\Brandrecognition2\env\Branddataset\Test'
excel_file = 'product_detection_results.xlsx'

# Run the bulk detection and update the Excel file
bulk_test_model(test_image_folder, excel_file)

