from django.urls import path
from . import views

urlpatterns = [
    path('files/', views.FileView.as_view(), name='file_list'),
    path('results/<str:pk>/', views.ResultView.as_view(), name='result_list')
]