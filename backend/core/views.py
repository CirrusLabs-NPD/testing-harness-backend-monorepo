from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .test_translation import test_user
import os
from django.conf import settings
from django.http import FileResponse

@api_view(['POST'])
def test_dataset(request):
    if request.method == 'POST':
        # Get selected models from request data
        selected_models = request.data['selected_models']
        # Get selected dataset from request data
        selected_dataset = request.data['dataset']

        # Call function to test selected models
        final_results, bleu_image, ter_image, meteor_image = test_user(selected_dataset, selected_models)

        bleu_image_url = os.path.join(settings.MEDIA_URL, os.path.basename(bleu_image))
        ter_image_url = os.path.join(settings.MEDIA_URL, os.path.basename(ter_image))
        meteor_image_url = os.path.join(settings.MEDIA_URL, os.path.basename(meteor_image))

        print(f"BLEU URL: {bleu_image_url}")
        print(f"METEOR URL: {meteor_image_url}")
        # Return JSON response with test results
        return Response({'selected_models': selected_models, 'selected_dataset': selected_dataset, 'final_results': final_results, 
                        'bleu_url': bleu_image_url, 'ter_url': ter_image_url, 'meteor_url': meteor_image_url})
    else:
        # Return error response if method is not allowed
        return Response({'error': 'Models cannot be run'}, status=405)