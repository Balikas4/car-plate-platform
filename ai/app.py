from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import cv2
import os
import numpy as np
import easyocr
import onnxruntime as ort
import io
from PIL import Image
import numpy as np
from fastapi.staticfiles import StaticFiles
import re

app = FastAPI()

# Serve the 'crops' folder
app.mount("/crops", StaticFiles(directory="crops"), name="crops")

# Load the ONNX model
def load_model(model_path):
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    return session

# Preprocess the image for YOLOv5 ONNX model
def preprocess_image(image_bytes, input_size=(640, 640)):
    image = Image.open(io.BytesIO(image_bytes))
    image = np.array(image)
    original_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, input_size)
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_input = np.transpose(image_normalized, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    image_input = np.expand_dims(image_input, axis=0)  # Add batch dimension
    return image_input, original_image

# Run inference on the input image using the ONNX model
def run_inference(model, image_input):
    input_name = model.get_inputs()[0].name
    outputs = model.run(None, {input_name: image_input})
    return outputs

# Postprocess detections (NMS and scaling)
def postprocess_detections(outputs, input_shape, original_shape, confidence_threshold=0.6, iou_threshold=0.5):
    predictions = outputs[0]  # Model's output (batch_size, num_predictions, 85)
    detections = []

    for pred in predictions:
        for det in pred:
            confidence = det[4]  # Object confidence score
            if confidence > confidence_threshold:
                x_center, y_center, width, height = det[:4]
                x1 = int((x_center - width / 2) * original_shape[1] / input_shape[1])
                y1 = int((y_center - height / 2) * original_shape[0] / input_shape[0])
                x2 = int((x_center + width / 2) * original_shape[1] / input_shape[1])
                y2 = int((y_center + height / 2) * original_shape[0] / input_shape[0])
                detections.append([x1, y1, x2, y2, confidence])

    bboxes = np.array([d[:4] for d in detections], dtype=np.float32)
    scores = np.array([d[4] for d in detections], dtype=np.float32)
    indices = cv2.dnn.NMSBoxes(bboxes.tolist(), scores.tolist(), confidence_threshold, iou_threshold)

    if isinstance(indices, np.ndarray) and len(indices.shape) == 1:
        indices = [[i] for i in indices]

    filtered_detections = [detections[i[0]] for i in indices]
    return filtered_detections

# Perform OCR on the cropped image using EasyOCR
def perform_ocr(cropped_images):
    reader = easyocr.Reader(['en'])
    ocr_results = []
    for image_path in cropped_images:
        image = cv2.imread(image_path)
        text = reader.readtext(image)
        ocr_results.append(' '.join([item[1] for item in text]))  # Concatenate text
    return ocr_results

# Crop detected regions and save them
def crop_detections(image, detections, output_dir):
    cropped_images = []
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det[:4]
        crop = image[y1:y2, x1:x2]
        cropped_path = f"{output_dir}/crop_{i}.jpg"
        cv2.imwrite(cropped_path, crop)
        cropped_images.append(cropped_path)
    return cropped_images

# Function to clean the OCR'd license plate
def clean_license_plate(text):
    # Remove the "LT" combo explicitly
    text = text.replace("LT", "")
    
    # Remove any remaining non-alphanumeric characters
    cleaned_text = re.sub(r'[^A-Z0-9]', '', text)
    return cleaned_text

# Function to determine the license plate type
def determine_license_plate_type(plate_number):
    plate_patterns = {
        "Car": r"^[A-Z]{3}\d{3}$",       # ABC 123
        "Trailer": r"^[A-Z]{2}\d{3}$",   # AB 123
        "Moto": r"^\d{3}[A-Z]{2}$",      # 123 AB
        "Scooter": r"^\d{2}[A-Z]{3}$",   # 12 ABC
        "4-Wheel": r"^[A-Z]{2}\d{2}$"   # AB 12
    }
    
    for plate_type, pattern in plate_patterns.items():
        if re.match(pattern, plate_number):
            return plate_type
    
    return "Unknown"

# Modify the process_image endpoint to include cleaning and type determination
@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        # Load model and preprocess
        model_path = 'best.onnx'
        model = load_model(model_path)
        image_input, original_image = preprocess_image(image_bytes)
        input_shape = (640, 640)
        original_shape = original_image.shape[:2]

        # Run inference
        outputs = run_inference(model, image_input)
        detections = postprocess_detections(outputs, input_shape, original_shape)

        if len(detections) == 0:
            return JSONResponse(content={"message": "No detections found"}, status_code=400)

        # Crop images and perform OCR
        output_dir = "crops"
        os.makedirs(output_dir, exist_ok=True)
        cropped_images = crop_detections(original_image, detections, output_dir)
        ocr_results = perform_ocr(cropped_images)

        # Clean OCR results and determine license plate type
        cleaned_plates = []
        plate_types = []
        for ocr_text in ocr_results:
            cleaned_plate = clean_license_plate(ocr_text)
            plate_type = determine_license_plate_type(cleaned_plate)
            cleaned_plates.append(cleaned_plate)
            plate_types.append(plate_type)

        # Return results
        response_data = {
            "cropped_images": [f"http://localhost:8001/crops/{os.path.basename(path)}" for path in cropped_images],
            "ocr_results": cleaned_plates,
            "plate_types": plate_types
        }
        return JSONResponse(content=jsonable_encoder(response_data))

    except Exception as e:
        return JSONResponse(content={"message": str(e)}, status_code=500)

