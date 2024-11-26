from django.shortcuts import render

from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.http import JsonResponse
from MedicalBot.ROI import visualize_areas_of_interest
import torch
from MedicalBot.CompactViosn import LightEfficientMedicalNet
from django.conf import settings
import os
def signup(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']

        # Check if passwords match
        if password1 != password2:
            messages.error(request, "Passwords do not match!")
            return redirect('signup')

        # Check if username exists
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists!")
            return redirect('signup')

        # Check if email exists
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already exists!")
            return redirect('signup')

        # Create user
        try:
            user = User.objects.create_user(username=username, email=email, password=password1)
            user.save()
            messages.success(request, "Account created successfully!")
            return redirect('login')
        except Exception as e:
            messages.error(request, "An error occurred during registration!")
            return redirect('signup')

    return render(request, 'signup.html')

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            return redirect('home')  # Redirect to home page after successful login
        else:
            messages.error(request, "Invalid username or password!")
            return redirect('login')
            
    return render(request, 'login.html')

def home(request):
    return render(request, 'home.html')

import base64
from django.http import JsonResponse
from django.shortcuts import render

model = LightEfficientMedicalNet(num_classes=5,in_channels=1)
path = os.path.join(settings.BASE_DIR, 'MedicalBot/models/best_model.pth')
# Load best model
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])
device = torch.device("cuda")
model = model.to(device)  # Move model to GPU

def inference(request):
    if request.method == 'POST':    
        image = request.FILES['image']
        # Save the uploaded file temporarily
        temp_path = '/tmp/temp_image.jpg'  # Or use tempfile module for better handling
        with open(temp_path, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)
        # Pass the path instead of the file object
        result = visualize_areas_of_interest(model, temp_path, device)
        # Clean up
        os.remove(temp_path)
        confidence, predicted_class = result['prediction_probabilities'],result['predicted_class']
        response_data = {
            'text': f"{predicted_class} with {confidence[predicted_class]*100:.2f}% confidence",
            'image_base64': result['image_base64'],
        }
        return JsonResponse(response_data)
    return render(request, 'inference.html')

