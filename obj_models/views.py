import sys
from subprocess import run, PIPE
from django.shortcuts import render, redirect
from .forms import ImageUploadForm
from .models import Image
import subprocess
from django.http import HttpResponse

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('show_image', image_id=form.instance.id)
    else:
        form = ImageUploadForm()
    return render(request, 'upload_image.html', {'form': form})

def show_image(request, image_id):
    image = Image.objects.get(pk=image_id)
    return render(request, 'show_image.html', {'image': image})

def dataset(request):
    return render(request, 'dataset.html')

def models(request):
    return render(request, 'models.html')

def run_model(request, model_name):
    try:
        # Execute the respective Python script based on the selected model
        if model_name == 'Yolov8':
            metrics = subprocess.run([sys.executable, 'yolo_m.py'], capture_output=True)
        elif model_name == 'ResNet':
            metrics = subprocess.run(['python', 'resnet_script.py'], capture_output=True)
        elif model_name == 'Fasterrcnn':
            metrics = subprocess.run(['python', 'fasterrcnn.py'], capture_output=True)
        else:
            return HttpResponse(f'Model "{model_name}" not found.')

        # Check if the subprocess ran successfully
        if metrics.returncode != 0:
            raise subprocess.CalledProcessError(metrics.returncode, " ".join(metrics.args))

        # Decode the output from bytes to string
        metrics_output = metrics.stdout.decode()

        # Display the output of the Python script
        return render(request, 'model_output.html', {'metrics': metrics_output})
    except Exception as e:
        # Handle any exceptions and display the error
        return HttpResponse(f"An error occurred: {str(e)}")
