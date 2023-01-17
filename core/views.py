from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
import json
import os
import torch
import torch.nn as nn
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import File
from .serializers import FileSerializer
from torchvision import models
from torchvision import transforms
from PIL import Image
from django.conf import settings
import io


def CNN_Model(pretrained=True):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model = models.densenet121(pretrained=pretrained)
    num_filters = model.classifier.in_features
    model.classifier = nn.Linear(num_filters, 13)
    model = model.to(device)
    return model


MODEL_PATH = os.path.join(settings.STATIC_ROOT, "Plant-Disease-differentiator.pth")
model_final = CNN_Model(pretrained=False)
model_final.to(torch.device('cpu'))
model_final.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model_final.eval()
json_path = os.path.join(settings.STATIC_ROOT, "classes.json")
imagenet_mapping = json.load(open(json_path))
mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]


def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_nums, std=std_nums)
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model_final.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    class_name, human_label = imagenet_mapping[predicted_idx]
    return human_label


class ResultView(APIView):
    def get(self, request, pk, *args, **kwargs):
        image = File.objects.get(id=pk)
        result_list = []
        image_bytes = image.file.read()
        try:
            predicted_label = get_prediction(image_bytes)
            description = f"Plant-disease : {predicted_label}"
            result_list.append({'Disease found in the plant': predicted_label, 'desc': description})
        except RuntimeError as re:
            print(re)
        return Response(result_list)


class FileView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        image = File.objects.all()
        serializer = FileSerializer(image, many=True)
        return Response(serializer.data)

    def post(self, request, *args, **kwargs):
        file_serializer = FileSerializer(data=request.data)
        if file_serializer.is_valid():
            file_serializer.save()
            return Response(file_serializer.data, status=status.HTTP_201_CREATED)
        else:
            print('error', file_serializer.errors)
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


