import requests

url = "http://localhost:8000/process-image/"

# Path to the image file you want to test
image_path = "/home/balikas4/auto_numeriai/local_vps/ai/test_image.jpg"

# Open the image in binary mode and send it in a POST request
with open(image_path, "rb") as image_file:
    files = {"file": ("image.jpg", image_file, "image/jpeg")}
    response = requests.post(url, files=files)

# Print the response from the FastAPI server
print(response.json())
