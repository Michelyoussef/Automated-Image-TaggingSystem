import boto3
import json
#from sagemaker.pytorch import PyTorchModel  # No need for this
from sagemaker import get_execution_role

def create_lambda_function(function_name, role, handler, runtime, code):
    lambda_client = boto3.client('lambda')
    response = lambda_client.create_function(
        FunctionName=function_name,
        Runtime=runtime,
        Role=role,
        Handler=handler,
        Code={'ZipFile': code},
        Timeout=30
    )
    return response['FunctionArn']

def create_api_gateway(api_name, lambda_arn):
    api_client = boto3.client('apigatewayv2')
    
    # Create API
    api = api_client.create_api(
        Name=api_name,
        ProtocolType='HTTP'
    )
    
    # Create integration
    integration = api_client.create_integration(
        ApiId=api['ApiId'],
        IntegrationType='AWS_PROXY',
        IntegrationMethod='POST',
        PayloadFormatVersion='2.0',
        IntegrationUri=lambda_arn
    )
    
    # Create route
    api_client.create_route(
        ApiId=api['ApiId'],
        RouteKey='POST /predict',
        Target=f'integrations/{integration["IntegrationId"]}'
    )
    
    # Deploy API
    api_client.create_deployment(
        ApiId=api['ApiId'],
        StageName='prod'
    )
    
    return api['ApiId']

def main():
    sagemaker_role = get_execution_role()

    # Create Lambda function
    lambda_role = 'arn:aws:iam::495599732227:role/service-role/coco-image-tagging-lambda-role-x2iv0bt7'
    with open('lambda_function.py', 'rb') as f:
        lambda_code = f.read()
    lambda_arn = create_lambda_function(
        'coco-image-tagging-lambda', lambda_role, 'lambda_function.lambda_handler', 'python3.8', lambda_code
    )

    
    api_id = create_api_gateway('coco-image-tagging-api', lambda_arn)

    print(f"API Gateway created with ID: {api_id}")

if __name__ == "__main__":
    main()