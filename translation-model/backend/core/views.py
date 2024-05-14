from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .main import test_custom
import os
from django.conf import settings
from django.http import HttpResponse

@api_view(['POST'])
def test_dataset(request):
    if request.method == 'POST':
        # Get selected models from request data
        selected_models = request.data['selected_models']
        # Get selected dataset from request data
        selected_datasets = request.data['datasets']

        if not selected_models or not selected_datasets:
            return HttpResponse(status=400)

        # response_data = {}
        
        # # Function to send SSE updates
        # class TestSSEView(BaseSseView):
        #     def iterator(self):
        #         nonlocal response_data

        #         # Call function to test selected models
        #         results_generator = test_user(selected_datasets, selected_models)

        #         for result in results_generator:
        #             bleu_image_url = os.path.join(settings.MEDIA_URL, os.path.basename(result['bleu_image']))
        #             ter_image_url = os.path.join(settings.MEDIA_URL, os.path.basename(result['ter_image']))
        #             meteor_image_url = os.path.join(settings.MEDIA_URL, os.path.basename(result['meteor_image']))
        #             accuracy_image_url = os.path.join(settings.MEDIA_URL, os.path.basename(result['accuracy_image']))
                    
        #             response_data['bleu_image'] = bleu_image_url
        #             response_data['ter_image'] = ter_image_url
        #             response_data['meteor_image'] = meteor_image_url
        #             response_data['accuracy_image'] = accuracy_image_url
        #             yield self.sse.add_message('update', response_data)

        #         # Close the connection when processing is complete
        #         yield self.sse.close()

        # return TestSSEView.as_view()(request)

        # Call function to test selected models
        bleu_image, ter_image, accuracy_image = test_custom(selected_datasets, selected_models)
        
        bleu_image_url = os.path.join(settings.MEDIA_URL, os.path.basename(bleu_image))
        ter_image_url = os.path.join(settings.MEDIA_URL, os.path.basename(ter_image))
        # meteor_image_url = os.path.join(settings.MEDIA_URL, os.path.basename(meteor_image))
        accuracy_image_url = os.path.join(settings.MEDIA_URL, os.path.basename(accuracy_image))
        
        # Return JSON response with test results
        return Response({'bleu_url': bleu_image_url, 'ter_url': ter_image_url, 'accuracy_url': accuracy_image_url})
    else:
        # Return error response if method is not allowed
        return Response({'error': 'Models cannot be run'}, status=405)