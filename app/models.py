from django.db import models

# Create your models here.
class CSV_Data(models.Model):
    trained_model_name = models.CharField(max_length = 100)
    label_encoder_model_name = models.CharField(max_length = 100)
    feature_names = models.CharField(max_length = 800)

    def __str__(self):
        return self.trained_model_name