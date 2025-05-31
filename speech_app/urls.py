from django.urls import path
from . import views

from .views import home, upload_audio,save_recorded_audio

urlpatterns = [
    path('home/', views.home, name='home'),
    path('upload/', upload_audio, name='upload_audio'),
    path("upload-recording/", save_recorded_audio, name="upload_recording"),
    path('signup/', views.signup, name='signup'),  # Add this line


]
