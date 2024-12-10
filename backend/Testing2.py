from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import json
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load the saved model
model_path = 'my_model.keras'
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

# Load the class indices (mapping from class labels to indices)
class_indices_path = 'class_indices.json'
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)
label_map = {v: k for k, v in class_indices.items()}

# Preprocess the image for prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image to input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Ensure the uploads directory exists
os.makedirs("uploads", exist_ok=True)

# File path for the Excel sheet
excel_file_path = "product_predictions.xlsx"

# Create or load the Excel file
def initialize_excel():
    if not os.path.exists(excel_file_path):
        df = pd.DataFrame(columns=['S. No.', 'File Name', 'Predicted Brand', 'Timestamp'])
        df.to_excel(excel_file_path, index=False)

# Function to update Excel with new data
def update_excel(data):
    if os.path.exists(excel_file_path):
        df = pd.read_excel(excel_file_path)
    else:
        df = pd.DataFrame(columns=['S. No.', 'File Name', 'Predicted Brand', 'Timestamp'])

    new_row = pd.DataFrame(data, columns=df.columns)
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_excel(excel_file_path, index=False)

# Prediction API
@app.route('/predict-folder', methods=['POST'])
def predict_folder():
    if 'images' not in request.files:
        return jsonify({"error": "No files found in the request"}), 400

    images = request.files.getlist('images')
    results = []
    excel_data = []

    # Loop through each file
    for idx, img_file in enumerate(images, start=1):
        filename = secure_filename(img_file.filename)
        file_path = os.path.join('uploads', filename)
        img_file.save(file_path)

        # Preprocess the image
        img_array = preprocess_image(file_path)

        # Predict using the model
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        predicted_label = label_map[predicted_class]

        # Get current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Prediction result
        result = {
            'predicted_brand': predicted_label

        }
        results.append(result)

        # Prepare data for Excel
        excel_data.append([
            len(results),  # S. No.
            filename,  # File Name
            predicted_label,  # Predicted Brand
            current_time  # Timestamp
        ])

        # Clean up the uploaded file (optional)
        os.remove(file_path)

    # Update Excel
    update_excel(excel_data)

    return jsonify(results)

if __name__ == '__main__':
    initialize_excel()
    app.run(debug=True)








# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import os
# import json
# import pandas as pd
# from datetime import datetime

# # Load the saved model
# model_path = 'my_model.keras'
# try:
#     model = load_model(model_path)
#     print("Model loaded successfully!")
# except Exception as e:
#     print(f"Failed to load model. Error: {e}")
#     exit()

# # Load the class indices (the mapping from class labels to indices)
# class_indices_path = 'class_indices.json'
# try:
#     with open(class_indices_path, 'r') as f:
#         class_indices = json.load(f)
#         print("Class indices loaded successfully!")
# except Exception as e:
#     print(f"Failed to load class indices. Error: {e}")
#     exit()

# # Reverse the mapping to go from index to class label
# label_map = {v: k for k, v in class_indices.items()}

# # Function to preprocess a single image
# def preprocess_image(img_path):
#     try:
#         img = image.load_img(img_path, target_size=(224, 224))  # Resize image to input size
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#         img_array /= 255.0  # Normalize pixel values
#         return img_array
#     except Exception as e:
#         print(f"Error in preprocessing image {img_path}. Error: {e}")
#         return None

# # Function to predict and update results in an Excel file
# def bulk_test_model(image_folder, excel_file):
#     # Check if the folder exists
#     if not os.path.exists(image_folder):
#         print(f"Folder does not exist: {image_folder}")
#         return

#     # List all image files in the folder
#     image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

#     if not image_files:
#         print(f"No image files found in the folder: {image_folder}")
#         return

#     # Check if the Excel file already exists
#     if os.path.exists(excel_file):
#         df = pd.read_excel(excel_file)
#     else:
#         df = pd.DataFrame(columns=['S. No.', 'Product Name', 'Time', 'Count'])

#     # Process each image file
#     for img_name in image_files:
#         img_path = os.path.join(image_folder, img_name)
#         print(f"Processing image: {img_name}")

#         # Preprocess the image
#         img_array = preprocess_image(img_path)
#         if img_array is None:
#             continue

#         # Predict the brand
#         predictions = model.predict(img_array)
#         predicted_class = np.argmax(predictions[0])
#         predicted_label = label_map[predicted_class]

#         # Get current time
#         current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#         # Check if the product is already in the Excel file
#         if predicted_label in df['Product Name'].values:
#             df.loc[df['Product Name'] == predicted_label, 'Count'] += 1
#             df.loc[df['Product Name'] == predicted_label, 'Time'] = current_time
#         else:
#             new_row = pd.DataFrame({
#                 'S. No.': [len(df) + 1],
#                 'Product Name': [predicted_label],
#                 'Time': [current_time],
#                 'Count': [1]
#             })
#             df = pd.concat([df, new_row], ignore_index=True)

#         # Display the prediction result
#         print(f"Image: {img_name}")
#         print(f"Predicted Brand: {predicted_label}")
#         print(f"Confidence: {np.max(predictions[0]) * 100:.2f}%")
#         print("-" * 50)

#     # Save the updated dataframe to Excel
#     df.to_excel(excel_file, index=False)
#     print(f"Results updated in {excel_file}")

# # Specify the folder containing the test images and the Excel file to update
# test_image_folder = r'C:\Users\uvara\OneDrive\Desktop\lets begin\Brandrecognition2\env\Branddataset\Test'
# excel_file = 'product_detection_results.xlsx'

# # Run the bulk detection and update the Excel file
# bulk_test_model(test_image_folder, excel_file)

