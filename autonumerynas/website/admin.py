from django.contrib import admin
from .models import PlateLT

@admin.register(PlateLT)
class PlateLTAdmin(admin.ModelAdmin):
    # Fields to display in the admin list view
    list_display = ('plate_number', 'plate_type', 'price', 'status', 'listed_by', 'created_at', 'updated_at')

    # Fields to search in the admin search box
    search_fields = ('plate_number', 'listed_by__username')

    # Filters for the admin list view
    list_filter = ('plate_type', 'status', 'created_at', 'updated_at')

    # Ordering of the records in the list view
    ordering = ('-created_at',)  # Show most recently created records first

    # Read-only fields (cannot be edited directly in the admin form)
    readonly_fields = ('id', 'created_at', 'updated_at')

    # Fields to show in the detailed admin form view
    fieldsets = (
        ('Plate Information', {
            'fields': ('plate_number', 'plate_type', 'image', 'price', 'status')
        }),
        ('Owner Information', {
            'fields': ('listed_by',)
        }),
        ('Timestamps', {
            'fields': ('id', 'created_at', 'updated_at'),
        }),
    )

    # Custom behavior for saving the object
    def save_model(self, request, obj, form, change):
        if not obj.listed_by:  # If listed_by is not set, assign the current user
            obj.listed_by = request.user
        super().save_model(request, obj, form, change)
