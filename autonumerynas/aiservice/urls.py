from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_image, name='upload_image'),
    path('save-license-plates/', views.save_license_plates, name='save_license_plates'),
]
