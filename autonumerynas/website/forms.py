from django import forms
from .models import PlateLT
import re

class PlateLTForm(forms.ModelForm):
    class Meta:
        model = PlateLT
        fields = ['plate_number', 'plate_type', 'image', 'price', 'status']
        widgets = {
            'plate_number': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., ABC 123 for cars',
            }),
            'plate_type': forms.Select(attrs={
                'class': 'form-select',
            }),
            'price': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter price',
            }),
            'status': forms.Select(attrs={
                'class': 'form-select',
            }),
            'image': forms.ClearableFileInput(attrs={
                'class': 'form-control',
            }),
        }

    def clean_plate_number(self):
        # Validate the plate_number format
        plate_number = self.cleaned_data.get('plate_number')
        plate_type = self.cleaned_data.get('plate_type')

        format_patterns = {
            'car': r'^[A-Z]{3}\d{3}$',             # ABC 123
            'trailer': r'^[A-Z]{2}\d{3}$',         # AB 123
            'moto': r'^\d{3}[A-Z]{2}$',            # 123 AB
            'scooter': r'^\d{2}[A-Z]{3}$',         # 12 ABC
            '4wheel': r'^[A-Z]{2}\d{2}$',          # AB 12
        }

        if plate_type in format_patterns:
            pattern = format_patterns[plate_type]
            if not re.match(pattern, plate_number):
                raise forms.ValidationError(f"Invalid format for {plate_type} plate.")

        return plate_number
