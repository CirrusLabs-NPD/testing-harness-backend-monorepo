# Welcome to testing-harness-backend monorepo

This repository containes the backends for the [testing-harness](https://github.com/CirrusLabs-NPD/testing-harness-dashboard) project.

You can navigate to each section to explore more about the services and their implementations.

# Django Project: Object Detection Web Application

## Prerequisites
Before starting, ensure you have the following installed on your system:
- Python 3.8 or later
- pip (Python package installer)
- absl-py==2.0.0
- arrow==1.3.0
- asgiref==3.8.0
- asttokens==2.4.1
- astunparse==1.6.3
- beautifulsoup4==4.12.2
- blinker==1.6.3 
- cachetools==5.3.1
- certifi==2024.2.2
- charset-normalizer==3.3.2
- click==8.1.7
- colorama==0.4.6
- comm==0.2.1
- contourpy==1.2.0
- cycler==0.12.1
- debugpy==1.8.1
- decorator==5.1.1
- deepface==0.0.79
- Django==5.0.3
- django-cors-headers==4.3.1
- djangorestframework==3.15.1
- executing==2.0.1
- filelock==3.13.1
- fire==0.5.0
- Flask==3.0.0
- Flask-Cors==4.0.0
- flatbuffers==23.5.26
- fonttools==4.49.0
- fsspec==2024.2.0
- gast==0.5.4
- gdown==4.7.1
- google-auth==2.23.3
- google-auth-oauthlib==1.0.0
- google-pasta==0.2.0
- grpcio==1.59.0
- gunicorn==21.2.0
- h5py==3.10.0
- idna==3.6
- ipykernel==6.29.2
- ipython==8.22.1
- itsdangerous==2.1.2
- jedi==0.19.1
- Jinja2==3.1.3
- jupyter_client==8.6.0
- jupyter_core==5.7.1
- keras==2.14.0
- kiwisolver==1.4.5
- libclang==16.0.6
- Markdown==3.5
- MarkupSafe==2.1.5
- matplotlib==3.8.3
- matplotlib-inline==0.1.6
- ml-dtypes==0.2.0
- mpmath==1.3.0
- mtcnn==0.1.1
- nest-asyncio==1.6.0
- networkx==3.2.1
- numpy==1.26.4
- oauthlib==3.2.2
- onnx==1.15.0
- opencv-python==4.9.0.80
- opt-einsum==3.3.0
- packaging==23.2
- pandas==2.2.1
- parso==0.8.3
- pillow==10.2.0
- platformdirs==4.2.0
- prompt-toolkit==3.0.43
- protobuf==4.24.4
- psutil==5.9.8
- pure-eval==0.2.2
- py-cpuinfo==9.0.0
- pyasn1==0.5.0
- pyasn1-modules==0.3.0
- pycocotools==2.0.7
- Pygments==2.17.2
- pyparsing==3.1.1
- PySocks==1.7.1
- python-dateutil==2.8.2
- python-decouple==3.8
- pytz==2024.1
- pywin32==306
- PyYAML==6.0.1
- pyzmq==25.1.2
- requests==2.31.0
- requests-oauthlib==1.3.1
- retina-face==0.0.13
- rsa==4.9
- scipy==1.12.0
- seaborn==0.13.2
- six==1.16.0
- soupsieve==2.5
- sqlparse==0.4.4
- stack-data==0.6.3
- sympy==1.12
- tensorboard==2.14.1
- tensorboard-data-server==0.7.1
- tensorflow==2.14.0
- tensorflow-estimator==2.14.0
- tensorflow-intel==2.14.0
- tensorflow-io-gcs-filesystem==0.31.0
- termcolor==2.3.0
- thop==0.1.1.post2209072238
- torch==2.2.1
- torchvision==0.17.1
- tornado==6.4
- tqdm==4.66.2
- traitlets==5.14.1
- types-python-dateutil==2.8.19.14
- typing_extensions==4.10.0
- tzdata==2024.1
- ultralytics==8.1.19
- urllib3==2.2.1
- wcwidth==0.2.13
- Werkzeug==3.0.0
- wrapt==1.14.1

- **Clone the repository**

-Ensure your settings.py is correctly configured.
-Add your development server's address to CORS_ALLOWED_ORIGINS if needed
-to run frontend type yarn dev and backend is py manage.py runserver 8080 

## Usage
-Uploading Images 
  -upload your images/labels for the test dataset into images and labels folder respectly and then select the script you want to test for output
-Navigate to the homepage (http://localhost:8000).
-Upload an image using the provided form.
-After uploading, you will be redirected to a page showing the uploaded image.

## Running Models
-Navigate to the /models/ page.
-Select a model and click the "Run Model" button.
-The results will be displayed, showing the detected objects and their metrics.

## Project Structure
-settings.py: Django project settings.
-urls.py: URL routing for the project.
  - models, dataset, run model, show output are the endpoints
-models.py: Database models. (images and labels folder)
-views.py: Views handling HTTP requests and responses. 
  - This is where the model integration is being done
  - Apporiate model scripts are link
-forms.py: Forms for image upload.
-obj_models/urls.py: URL routing for the obj_models app.
-detfasterrcnn.py: Script for running Faster R-CNN.
-yolo_m.py: Script for running YOLOv8.
-manage.py: Django management utility.