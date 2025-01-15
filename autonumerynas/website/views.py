from django.shortcuts import render, redirect
from .models import PlateLT
from django.contrib.auth.forms import UserCreationForm


# Home page view
def home(request):
    plates = PlateLT.objects.all()  # Fetch all license plates
    return render(request, 'home.html', {'plates': plates})

# Sell license plates view
def sell_license_plates(request):
    return render(request, 'sell_license_plates.html')

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')  # Redirect to login page after successful registration
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})
