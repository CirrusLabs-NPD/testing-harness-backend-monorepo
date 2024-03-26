from django.urls import path
from . import views

urlpatterns = [
    path('test-dataset/', views.test_dataset, name='test_dataset')
]