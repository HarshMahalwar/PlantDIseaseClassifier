from django.db import models


class File(models.Model):
    objects = None
    file = models.FileField(upload_to='images/')
    date_created = models.DateTimeField(auto_now_add=True)