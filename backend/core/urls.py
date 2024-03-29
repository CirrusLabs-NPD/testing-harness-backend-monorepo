from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('test_dataset/', views.test_dataset, name='test_dataset')
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)