import cv2
import os
import numpy as np
import onnxruntime as ort
import easyocr

def load_model(model_path):
    """
    Load the ONNX model using ONNX Runtime.
    """
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    return session

def preprocess_image(image_path, input_size=(640, 640)):
    """
    Preprocess the image for YOLOv5 ONNX model:
    - Resize to input size
    - Normalize pixel values
    - Add batch dimension
    - Rearrange channels to (batch_size, channels, height, width)
    """
    image = cv2.imread(image_path)
    original_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, input_size)
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_input = np.transpose(image_normalized, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    image_input = np.expand_dims(image_input, axis=0)  # Add batch dimension
    return image_input, original_image

def run_inference(model, image_input):
    """
    Run inference on the input image using the ONNX model.
    """
    input_name = model.get_inputs()[0].name
    outputs = model.run(None, {input_name: image_input})
    return outputs

def postprocess_detections(outputs, input_shape, original_shape, confidence_threshold=0.6, iou_threshold=0.5):
    """
    Postprocess YOLOv5 ONNX model outputs:
    - Apply confidence threshold
    - Scale boxes back to the original image size
    - Apply Non-Maximum Suppression (NMS)
    """
    predictions = outputs[0]  # Model's output (batch_size, num_predictions, 85)
    detections = []

    # YOLOv5 output: [batch_size, num_predictions, 85]
    for pred in predictions:
        for det in pred:
            confidence = det[4]  # Object confidence score
            if confidence > confidence_threshold:
                # Extract bounding box and rescale to original image size
                x_center, y_center, width, height = det[:4]
                x1 = int((x_center - width / 2) * original_shape[1] / input_shape[1])
                y1 = int((y_center - height / 2) * original_shape[0] / input_shape[0])
                x2 = int((x_center + width / 2) * original_shape[1] / input_shape[1])
                y2 = int((y_center + height / 2) * original_shape[0] / input_shape[0])

                # Class probabilities
                class_id = int(np.argmax(det[5:]))
                class_conf = det[5 + class_id]

                detections.append([x1, y1, x2, y2, confidence, class_id, class_conf])

    if not detections:
        print("No detections passed the confidence threshold.")
        return []

    # Prepare for NMS
    bboxes = np.array([d[:4] for d in detections], dtype=np.float32)
    scores = np.array([d[4] for d in detections], dtype=np.float32)

    if len(bboxes) == 0:
        print("No bounding boxes available for NMS.")
        return []

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(
        bboxes.tolist(), scores.tolist(), confidence_threshold, iou_threshold
    )

    # Ensure indices are iterable
    if isinstance(indices, np.ndarray) and len(indices.shape) == 1:
        indices = [[i] for i in indices]

    if len(indices) == 0:
        print("No detections remained after NMS.")
        return []

    # Filter detections based on NMS indices
    filtered_detections = [detections[i[0]] for i in indices]

    return filtered_detections



def draw_boxes(image, detections, class_names):
    """
    Draw bounding boxes on the image without adding class labels.
    """
    for det in detections:
        x1, y1, x2, y2, confidence, class_id, class_conf = det
        color = (0, 255, 0)  # Green bounding box
        # Just draw the bounding box without the label text
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image


def crop_detections(image, detections, output_dir):
    """Crop detected regions and save them as individual files."""
    cropped_images = []
    for i, det in enumerate(detections):
        x1, y1, x2, y2, _, _, _ = det  # Unpack detection
        crop = image[y1:y2, x1:x2]
        cropped_path = f"{output_dir}/crop_{i}.jpg"
        cv2.imwrite(cropped_path, crop)
        cropped_images.append(cropped_path)

    print(f"Saved {len(cropped_images)} cropped images to {output_dir}")
    return cropped_images

def perform_ocr(cropped_images):
    """Perform OCR using EasyOCR."""
    reader = easyocr.Reader(['en'])  # Initialize reader with the desired languages
    results = []
    for image_path in cropped_images:
        image = cv2.imread(image_path)
        text = reader.readtext(image)
        results.append(' '.join([item[1] for item in text]))  # Concatenate text from the bounding boxes
    return results

def main():
    # Paths to model, image, and output
    model_path = 'best.onnx'
    image_path = '/home/balikas4/auto_numeriai/plate_read/plate_imgs/image_12.jpg'
    output_image_path = 'output_image_with_boxes.jpg'
    output_crop_dir = 'crops'  # Directory to save cropped detections
    class_names = ["plate"]  # Replace with your class names

    # Create output directory for crops if it doesn't exist
    if not os.path.exists(output_crop_dir):
        os.makedirs(output_crop_dir)

    # Load ONNX model
    print("Loading model...")
    model = load_model(model_path)

    # Preprocess image
    print("Preprocessing image...")
    image_input, original_image = preprocess_image(image_path)
    input_shape = (640, 640)  # Model input size (H, W)
    original_shape = original_image.shape[:2]  # Original image size (H, W)

    # Run inference
    print("Running inference...")
    outputs = run_inference(model, image_input)
    print(f"Model outputs shape: {outputs[0].shape}")

    # Postprocess detections
    print("Postprocessing detections...")
    detections = postprocess_detections(outputs, input_shape, original_shape)

    if len(detections) == 0:
        print("No detections found.")
        return
    else:
        print(f"Found {len(detections)} detections.")

    # Draw bounding boxes
    print("Drawing bounding boxes...")
    output_image = draw_boxes(original_image, detections, class_names)

    # Save the output image with bounding boxes
    cv2.imwrite(output_image_path, output_image)
    print(f"Output image saved at {output_image_path}")

    # Crop detections and save them as individual files
    print("Cropping detections...")
    cropped_images = crop_detections(original_image, detections, output_crop_dir)
    print(f"Cropped images saved: {cropped_images}")

    # Perform OCR on the cropped images
    print("Performing OCR on cropped images...")
    ocr_results = perform_ocr(cropped_images)

    # Print the OCR results
    print("OCR Results:")
    for idx, text in enumerate(ocr_results):
        print(f"Cropped Image {idx+1}: {text}")
if __name__ == '__main__':
    main()
