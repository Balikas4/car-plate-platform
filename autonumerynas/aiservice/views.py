from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django import forms
import requests
import os
from website.models import PlateLT


# Define the form directly in the view file
class UploadImageForm(forms.Form):
    image = forms.ImageField()

AI_SERVICE_URL = "http://ai-service:8001/process-image/"  # Assuming the AI service is running locally

def upload_image(request):
    form = UploadImageForm()  # Initialize the form

    if request.method == 'POST' and request.FILES.get('image'):
        form = UploadImageForm(request.POST, request.FILES)  # Bind form with data
        if form.is_valid():
            # Save the uploaded image
            image = request.FILES['image']
            image_path = os.path.join("tmp", image.name)
            os.makedirs("tmp", exist_ok=True)  # Ensure the tmp directory exists
            with open(image_path, 'wb') as f:
                for chunk in image.chunks():
                    f.write(chunk)

            # Send the image to the AI service
            try:
                with open(image_path, 'rb') as img:
                    files = {'file': img}
                    response = requests.post(AI_SERVICE_URL, files=files)
                    response.raise_for_status()
            except requests.exceptions.RequestException as e:
                return JsonResponse({"error": f"AI service error: {str(e)}"}, status=500)

            # Process the response
            if response.status_code == 200:
                data = response.json()
                crops = data.get("cropped_images", [])
                ocr_results = data.get("ocr_results", [])

                # Combine the cropped image URLs and OCR results for rendering
                crops_and_results = zip(crops, ocr_results)

                # Render the review page
                return render(request, 'review_crops.html', {
                    "crops_and_results": crops_and_results
                })
            else:
                return JsonResponse({"error": "Error processing image"}, status=500)

    # Render the upload form if GET request or invalid form submission
    return render(request, 'upload_image.html', {'form': form})

def save_license_plates(request):
    if request.method == 'POST':
        # Get the edited license plates
        plates = request.POST.getlist('plates')  # List of plates
        plate_types = request.POST.getlist('plate_types')  # Optional, if you send plate types

        saved_plates = []
        for plate, plate_type in zip(plates, plate_types):
            # Ensure the plate is saved only if it's valid and unique
            if not PlateLT.objects.filter(plate_number=plate).exists():
                license_plate = PlateLT(
                    plate_number=plate,
                    plate_type=plate_type,
                )
                license_plate.save()
                saved_plates.append(license_plate)

        return JsonResponse({"message": "License plates saved successfully!", "saved_plates": [p.plate_number for p in saved_plates]})

    return JsonResponse({"error": "Invalid request"}, status=400)
