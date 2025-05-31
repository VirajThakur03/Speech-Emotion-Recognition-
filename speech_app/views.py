import os
from django.shortcuts import render,redirect
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.http import JsonResponse
from .forms import AudioUploadForm
from .models import AudioFile
from .ml_model import predict_emotion  
from tensorflow.keras.models import load_model
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login as auth_login
from django.contrib.auth import login
from .user_forms import SignUpForm



# Define model path
MODEL_PATH = os.path.join("speech_app", "bilstm_attention_model_ravdess_cremad.h5")

# Load model safely using keras
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = load_model(MODEL_PATH)


# Home view for displaying the form and emotion results
@login_required
def home(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        audio_instance = AudioFile(file=uploaded_file)
        audio_instance.save()

        file_path = audio_instance.file.path
        emotion = predict_emotion(file_path, model)

        return render(request, 'home.html', {'emotion': emotion})

    return render(request, 'home.html')

def custom_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
    else:
        form = AuthenticationForm()
    return render(request, 'registration/login.html', {'form': form})

def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')
    else:
        form = SignUpForm()
    return render(request, 'registration/signup.html', {'form': form})



# Upload audio (for both file upload and recorded audio)
def upload_audio(request):
    if request.method == "POST":
        audio_file = request.FILES.get('audio_file') or request.FILES.get('recorded_audio')

        if not audio_file:
            return render(request, 'home.html', {'error': 'No audio file provided'})

        file_path = default_storage.save(f"uploads/{audio_file.name}", ContentFile(audio_file.read()))
        full_file_path = default_storage.path(file_path)

        predicted_emotion = predict_emotion(full_file_path, model)

        return render(request, 'home.html', {'emotion': predicted_emotion, 'file_name': audio_file.name})

    return render(request, 'home.html')


# Save recorded audio to media/uploads/
def save_recorded_audio(request):
    if request.method == "POST" and request.FILES.get("audio"):
        audio_file = request.FILES["audio"]
        file_path = f"uploads/{audio_file.name}"

        file_name = default_storage.save(file_path, ContentFile(audio_file.read()))
        file_url = default_storage.url(file_name)

        return JsonResponse({"message": "File uploaded successfully!", "file_url": file_url})
    
    return JsonResponse({"error": "No file uploaded"}, status=400)
