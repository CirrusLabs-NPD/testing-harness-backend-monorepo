# from django.urls import path
#
# from . import views
#
# urlpatterns = [
#     path("", views.index, name="index"),
#
from django.urls import path
from . import views

urlpatterns = [
    path("", views.upload_image, name="upload_image"),
    path("image/<int:image_id>/", views.show_image, name="show_image"),
    path("dataset/", views.dataset, name="dataset"),
    path('models/', views.models, name='models'),
    path('run_model/<str:model_name>/', views.run_model, name='run_model'),
    path('show_output/', views.show_output, name='show_output')
]
