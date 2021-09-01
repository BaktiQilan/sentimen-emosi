from django.db import models

# Create your models here.
class SentimentModel(models.Model):
    Sentence = models.CharField(max_length=120)


class DatasetModel(models.Model):
    pesan = models.TextField()
    label = models.CharField(max_length=255)

    def __str__(self):
        return self.label

    class Meta:
        ordering = ["pesan"]

class NewDataModel(models.Model):
    pesan = models.TextField()

class PrediksiDataModel(models.Model):
    pesan = models.TextField()
    label = models.CharField(max_length=255)

