from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from test_translation import test_dataset, test_models
    
@api_view(['POST'])
def test_dataset(request):
    if request.method == 'POST':
        # Get selected models from request data
        selected_model = request.data.get('selected_model', [])
        # Get selected models from request data
        selected_dataset = request.data.get('selected_models', [])

        # Call function to test selected models
        test_results = test_models(selected_model)

        # Return JSON response with test results
        return Response(test_results)
    else:
        # Return error response if method is not allowed
        return Response({'error': 'Models cannot be run'}, status=405)