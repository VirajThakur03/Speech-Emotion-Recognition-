from django.db import models

class AudioFile(models.Model):
    file = models.FileField(upload_to="uploads/")  # Store audio file in 'uploads/' folder
    recorded_at = models.DateTimeField(auto_now_add=True)  # Automatically set timestamp on creation

class RecordedAudio(models.Model):
    file = models.FileField(upload_to="uploads/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Audio uploaded on {self.recorded_at}"
