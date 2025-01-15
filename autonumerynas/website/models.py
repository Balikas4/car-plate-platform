from django.db import models
from django.core.exceptions import ValidationError
import re

class PlateLT(models.Model):
    PLATE_TYPES = [
        ('car', 'Car'),
        ('trailer', 'Trailer'),
        ('moto', 'Moto'),
        ('scooter', 'Scooter'),
        ('4wheel', '4-Wheel'),
    ]

    id = models.UUIDField(primary_key=True, editable=False, unique=True)
    plate_number = models.CharField(max_length=10, db_index=True)
    plate_type = models.CharField(max_length=15, choices=PLATE_TYPES)
    image = models.ImageField(upload_to='plates/', null=True, blank=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    status = models.CharField(max_length=20, default='Available', choices=[
        ('Available', 'Available'),
        ('Sold', 'Sold'),
        ('Reserved', 'Reserved'),
    ])
    listed_by = models.ForeignKey('auth.User', on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def clean(self):
        # Validation for plate_number based on plate_type
        format_patterns = {
            'car': r'^[A-Z]{3} \d{3}$',             # ABC 123
            'trailer': r'^[A-Z]{2} \d{3}$',         # AB 123
            'moto': r'^\d{3} [A-Z]{2}$',            # 123 AB
            'scooter': r'^\d{2} [A-Z]{3}$',         # 12 ABC
            '4wheel': r'^[A-Z]{2} \d{2}$',          # AB 12
        }
        pattern = format_patterns.get(self.plate_type)
        if pattern and not re.match(pattern, self.plate_number):
            raise ValidationError(f"Invalid format for {self.plate_type} plate.")

    def __str__(self):
        return f"{self.plate_number} ({self.plate_type})"
