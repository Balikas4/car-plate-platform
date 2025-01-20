from django.shortcuts import render, redirect
from .models import PlateLT
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from .forms import PlateLTForm
from django.contrib import messages


# Home page view
def home(request):
    plates = PlateLT.objects.all()  # Fetch all license plates
    return render(request, 'home.html', {'plates': plates})

@login_required
def sell_license_plates(request):
    if request.method == 'POST':
        form = PlateLTForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the form instance, but don't commit yet
            license_plate = form.save(commit=False)
            license_plate.listed_by = request.user  # Associate the current user
            license_plate.save()
            messages.success(request, 'License plate listed successfully!')
            return redirect('home')  # Replace 'main_page' with the actual name of the URL for your main page
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = PlateLTForm()

    return render(request, 'sell_license_plates.html', {'form': form})


def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')  # Redirect to login page after successful registration
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})
