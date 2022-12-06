import boto3
import json


def invoke_endpoint(endpoint_name, message):
    '''Call the sagemaker (serverless) inference endpoint
    '''
    print("invoke_endpoint: endpoint_name: %s" % (endpoint_name))
    print("invoke_endpoint: message: %s" % (message))

    client = boto3.client('sagemaker-runtime')
    
    content_type = "application/json"
    # must specify "inputs"
    data = {
        "inputs": message
    }
    print("invoke_endpoint: data: %s" % json.dumps(data, indent=2))
    
    ie_response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType=content_type,
        Body=json.dumps(data)
    )
    # ie_response['Body'] is a byte stream so we can't format it
    print("invoke_endpoint: ie_response: %s" % (ie_response))

    # ic_response['Body'] is a byte stream so we need to read and decode it
    response = ie_response["Body"].read().decode("utf-8")
    print("invoke_endpoint: response: %s" % json.dumps(response, indent=2))

    return(response)

