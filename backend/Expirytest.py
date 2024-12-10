from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import easyocr
import json
import re
from datetime import datetime
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Load models and resources
yolo_model = YOLO('best1143images(100).pt')  # YOLO model for expiry detection
expiry_keras_model = load_model('date_detection_model1.keras')  # Keras model for expiry detection
brand_keras_model = load_model('my_model.keras')  # Keras model for brand detection

# Load class indices for brand detection
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Reverse the class indices mapping for decoding
class_indices_reversed = {v: k for k, v in class_indices.items()}

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Function to preprocess the image for Keras model
def preprocess_for_keras(image):
    image = cv2.resize(image, (224, 224))  # Resize to MobileNetV2 input size
    image = image.astype("float32") / 255.0  # Normalize pixel values
    image = img_to_array(image)  # Convert to array
    image = np.expand_dims(image, axis=0)  
    return image

# Function to detect the brand
def detect_brand(image_path):
    image = cv2.imread(image_path)
    preprocessed_image = preprocess_for_keras(image)
    predictions = brand_keras_model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    brand_name = class_indices_reversed.get(predicted_class, "Unknown")
    return brand_name

# Function to extract dates from text using regex
def extract_dates(text):
    date_patterns = [
        r'\b\d{2}/\d{2}/\d{4}\b',    # MM/DD/YYYY or DD/MM/YYYY
        r'\b\d{4}/\d{2}/\d{2}\b',    # YYYY/MM/DD
        r'\b\d{2}\.\d{2}\.\d{4}\b',  # MM.DD.YYYY or DD.MM.YYYY
        r'\b\d{4}\.\d{2}\.\d{2}\b',  # YYYY.MM.DD
        r'\b\d{2}-\d{2}-\d{4}\b',    # MM-DD-YYYY or DD-MM-YYYY
        r'\b\d{4}-\d{2}-\d{2}\b',    # YYYY-MM-DD
    ]
    detected_dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                detected_dates.append(parse_date(match))
            except ValueError:
                continue
    return detected_dates

# Function to parse date string to datetime object
def parse_date(date_str):
    date_formats = [
        "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
        "%m.%d.%Y", "%d.%m.%Y", "%Y.%m.%d",
        "%m-%d-%Y", "%d-%m-%Y", "%Y-%m-%d",
    ]
    for date_format in date_formats:
        try:
            return datetime.strptime(date_str, date_format)
        except ValueError:
            continue
    raise ValueError(f"Date format not recognized: {date_str}")

# Function to select the latest date
def get_latest_date(dates):
    if not dates:
        return None
    return max(dates)

# Function to compare dates and determine expiration status
def compare_dates(latest_date):
    current_date = datetime.now()

    if latest_date < current_date:
        expired_status = "Yes"
        status_message = f"Expired on {latest_date.strftime('%d-%m-%Y')}"
        life_span = "NA"
    else:
        expired_status = "No"
        status_message = f"Valid until {latest_date.strftime('%d-%m-%Y')}"
        life_span = (latest_date - current_date).days

    return expired_status, life_span, status_message

# Function to save details to the Excel file
def save_details_to_excel(time, brand_name, expiry_date, expired_status, life_span, output_file):
    if os.path.exists(output_file):
        df = pd.read_excel(output_file)
    else:
        df = pd.DataFrame(columns=['Sno', 'Time', 'BrandName', 'Expiry Date', 'Expired Status', 'Life Span'])

    # Add new entry to the DataFrame
    new_row = pd.DataFrame({
        'Sno': [len(df) + 1],
        'Time': [time],
        'BrandName': [brand_name],
        'Expiry Date': [expiry_date.strftime('%d-%m-%Y') if expiry_date else "NA"],
        'Expired Status': [expired_status],
        'Life Span': [life_span if life_span != "NA" else "NA"]
    })
    df = pd.concat([df, new_row], ignore_index=True)

    # Save to the Excel file
    df.to_excel(output_file, index=False)
    print(f"Details saved to {output_file}")

@app.route('/recognize-expiry-date', methods=['POST'])
def recognize_expiry_date():
    if 'front_image' not in request.files or 'back_image' not in request.files:
        return jsonify({"error": "Both front and back images are required"}), 400

    front_image = request.files['front_image']
    back_image = request.files['back_image']

    # Save temporary images
    front_path = "temp_front.jpg"
    back_path = "temp_back.jpg"
    front_image.save(front_path)
    back_image.save(back_path)

    output_excel_file = 'Expiry_Brand_Details3.xlsx'

    try:
        # Call the process_images_and_save_details function to process the images
        brand_name = detect_brand(front_path)
        expiry_image = cv2.imread(back_path)
        results = yolo_model(expiry_image)

        detected_dates = []
        for result in results:
            boxes = result.boxes.xyxy
            confidences = result.boxes.conf

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                detected_region = expiry_image[y1:y2, x1:x2]
                preprocessed_region = preprocess_for_keras(detected_region)
                prediction = expiry_keras_model.predict(preprocessed_region)[0][0]

                if prediction < 0.5:
                    ocr_results = reader.readtext(detected_region)
                    for _, text, _ in ocr_results:
                        detected_dates.extend(extract_dates(text))

        # Determine the latest date
        latest_date = get_latest_date(detected_dates)
        expired_status, life_span, status_message = compare_dates(latest_date)

        # Save to Excel
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        save_details_to_excel(current_time, brand_name, latest_date, expired_status, life_span, output_excel_file)

        # Construct response
        response = {
            "brand_name": brand_name,
            "expiry_date": latest_date.strftime('%d-%m-%Y') if latest_date else "Not Detected",
            "expired_status": expired_status,
            "life_span": life_span if life_span != "NA" else "Not Applicable",
            "status_message": status_message
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Cleanup temporary files
        os.remove(front_path)
        os.remove(back_path)

if __name__ == '__main__':
    app.run(debug=True)




# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# import easyocr
# import json
# import re
# from datetime import datetime
# import pandas as pd
# import os

# app = Flask(__name__)
# CORS(app)

# # Load models and resources
# yolo_model = YOLO('best1143images(100).pt')  # YOLO model for expiry detection
# expiry_keras_model = load_model('date_detection_model1.keras')  # Keras model for expiry detection
# brand_keras_model = load_model('my_model.keras')  # Keras model for brand detection

# # Load class indices for brand detection
# with open('class_indices.json', 'r') as f:
#     class_indices = json.load(f)

# # Reverse the class indices mapping for decoding
# class_indices_reversed = {v: k for k, v in class_indices.items()}

# # Initialize EasyOCR reader
# reader = easyocr.Reader(['en'])

# # Function to preprocess the image for Keras model
# def preprocess_for_keras(image):
#     image = cv2.resize(image, (224, 224))  # Resize to MobileNetV2 input size
#     image = image.astype("float32") / 255.0  # Normalize pixel values
#     image = img_to_array(image)  # Convert to array
#     image = np.expand_dims(image, axis=0)  
#     return image

# # Function to detect the brand
# def detect_brand(image_path):
#     image = cv2.imread(image_path)
#     preprocessed_image = preprocess_for_keras(image)
#     predictions = brand_keras_model.predict(preprocessed_image)
#     predicted_class = np.argmax(predictions, axis=1)[0]
#     brand_name = class_indices_reversed.get(predicted_class, "Unknown")
#     return brand_name

# # Function to extract dates from text using regex
# def extract_dates(text):
#     date_patterns = [
#         r'\b\d{2}/\d{2}/\d{4}\b',    # MM/DD/YYYY or DD/MM/YYYY
#         r'\b\d{4}/\d{2}/\d{2}\b',    # YYYY/MM/DD
#         r'\b\d{2}\.\d{2}\.\d{4}\b',  # MM.DD.YYYY or DD.MM.YYYY
#         r'\b\d{4}\.\d{2}\.\d{2}\b',  # YYYY.MM.DD
#         r'\b\d{2}-\d{2}-\d{4}\b',    # MM-DD-YYYY or DD-MM-YYYY
#         r'\b\d{4}-\d{2}-\d{2}\b',    # YYYY-MM-DD
#     ]
#     detected_dates = []
#     for pattern in date_patterns:
#         matches = re.findall(pattern, text)
#         for match in matches:
#             try:
#                 detected_dates.append(parse_date(match))
#             except ValueError:
#                 continue
#     return detected_dates

# # Function to parse date string to datetime object
# def parse_date(date_str):
#     date_formats = [
#         "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
#         "%m.%d.%Y", "%d.%m.%Y", "%Y.%m.%d",
#         "%m-%d-%Y", "%d-%m-%Y", "%Y-%m-%d",
#     ]
#     for date_format in date_formats:
#         try:
#             return datetime.strptime(date_str, date_format)
#         except ValueError:
#             continue
#     raise ValueError(f"Date format not recognized: {date_str}")

# # Function to select the latest date
# def get_latest_date(dates):
#     if not dates:
#         return None
#     return max(dates)

# # Function to compare dates and determine expiration status
# def compare_dates(latest_date):
#     current_date = datetime.now()

#     if latest_date < current_date:
#         expired_status = "Yes"
#         status_message = f"Expired on {latest_date.strftime('%d-%m-%Y')}"
#         life_span = "NA"
#     else:
#         expired_status = "No"
#         status_message = f"Valid until {latest_date.strftime('%d-%m-%Y')}"
#         life_span = (latest_date - current_date).days

#     return expired_status, life_span, status_message

# # Function to save details to the Excel file
# def save_details_to_excel(time, brand_name, expiry_date, expired_status, life_span, output_file):
#     if os.path.exists(output_file):
#         df = pd.read_excel(output_file)
#     else:
#         df = pd.DataFrame(columns=['Sno', 'Time', 'BrandName', 'Expiry Date', 'Expired Status', 'Life Span'])

#     # Add new entry to the DataFrame
#     new_row = pd.DataFrame({
#         'Sno': [len(df) + 1],
#         'Time': [time],
#         'BrandName': [brand_name],
#         'Expiry Date': [expiry_date.strftime('%d-%m-%Y') if expiry_date else "NA"],
#         'Expired Status': [expired_status],
#         'Life Span': [life_span if life_span != "NA" else "NA"]
#     })
#     df = pd.concat([df, new_row], ignore_index=True)

#     # Save to the Excel file
#     df.to_excel(output_file, index=False)
#     print(f"Details saved to {output_file}")

# # Main function to process images and update details
# def process_images_and_save_details(expiry_image_path, brand_image_path, output_file):
#     # Detect brand
#     brand_name = detect_brand(brand_image_path)

#     # Detect expiry date
#     expiry_image = cv2.imread(expiry_image_path)
#     results = yolo_model(expiry_image)

#     detected_dates = []
#     for result in results:
#         boxes = result.boxes.xyxy
#         confidences = result.boxes.conf

#         for i, box in enumerate(boxes):
#             x1, y1, x2, y2 = map(int, box)
#             detected_region = expiry_image[y1:y2, x1:x2]

#             preprocessed_region = preprocess_for_keras(detected_region)
#             prediction = expiry_keras_model.predict(preprocessed_region)[0][0]

#             if prediction < 0.5:
#                 ocr_results = reader.readtext(detected_region)
#                 for _, text, _ in ocr_results:
#                     detected_dates.extend(extract_dates(text))

#     # Select the latest date
#     latest_date = get_latest_date(detected_dates)

#     # Determine expiration status and life span
#     expired_status, life_span, status_message = compare_dates(latest_date)

#     # Save to Excel
#     current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     save_details_to_excel(current_time, brand_name, latest_date, expired_status, life_span, output_file)

# # Specify paths for input images and output Excel file
# brand_image_path = r'C:\Users\uvara\OneDrive\Desktop\letexpiry_image_path = r'C:\Users\uvara\OneDrive\Desktop\lets begin\Flipkart product analysis\test_00149.jpg'  # Replace with expiry image path
# s begin\Flipkart product analysis\Brand Detection\env\Branddataset\Train\MysoreSandal\IMG20240914084249.jpg'    # Replace with brand image path
# output_excel_file = 'Expiry_Brand_Details3.xlsx'

# # Process images and save details
# process_images_and_save_details(expiry_image_path, brand_image_path, output_excel_file)



# @app.route('/recognize-expiry-date', methods=['POST'])
# def recognize_expiry_date():
#     if 'front_image' not in request.files or 'back_image' not in request.files:
#         return jsonify({"error": "Both front and back images are required"}), 400

#     front_image = request.files['front_image']
#     back_image = request.files['back_image']

#     # Save temporary images
#     front_path = "temp_front.jpg"
#     back_path = "temp_back.jpg"
#     front_image.save(front_path)
#     back_image.save(back_path)

#     try:
#         # Call the process_images_and_save_details function to process the images
#         brand_name = detect_brand(front_path)
#         expiry_image = cv2.imread(back_path)
#         results = yolo_model(expiry_image)

#         detected_dates = []
#         for result in results:
#             boxes = result.boxes.xyxy
#             confidences = result.boxes.conf

#             for i, box in enumerate(boxes):
#                 x1, y1, x2, y2 = map(int, box)
#                 detected_region = expiry_image[y1:y2, x1:x2]
#                 preprocessed_region = preprocess_for_keras(detected_region)
#                 prediction = expiry_keras_model.predict(preprocessed_region)[0][0]

#                 if prediction < 0.5:
#                     ocr_results = reader.readtext(detected_region)
#                     for _, text, _ in ocr_results:
#                         detected_dates.extend(extract_dates(text))

#         # Determine the latest date
#         latest_date = get_latest_date(detected_dates)
#         expired_status, life_span, status_message = compare_dates(latest_date)

#         # Construct response
#         response = {
#             "brand_name": brand_name,
#             "expiry_date": latest_date.strftime('%d-%m-%Y') if latest_date else "Not Detected",
#             "expired_status": expired_status,
#             "life_span": life_span if life_span != "NA" else "Not Applicable",
#             "status_message": status_message
#         }

#         return jsonify(response)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

#     finally:
#         # Cleanup temporary files
#         os.remove(front_path)
#         os.remove(back_path)

# if __name__ == '__main__':
#     app.run(debug=True)
