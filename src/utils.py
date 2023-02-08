from difflib import SequenceMatcher
import json
import re

from json import JSONDecoder

# Download an object from S3 as bytes
def _download_from_s3(bucket_name, key_name, client):
    response = client.get_object(Bucket=bucket_name, Key=key_name)['Body']
    return response.read()

# Download an object from S3 as JSON
def _download_json_from_s3(bucket_name, key_name, client):
    response = _download_from_s3(bucket_name, key_name, client)
    response_json = json.loads(response.decode())
    return response_json

# Read stacked json files
NOT_WHITESPACE = re.compile(r'\S')
def _decode_stacked(document, pos=0, decoder=JSONDecoder()):
    while True:
        match = NOT_WHITESPACE.search(document, pos)
        if not match:
            return
        pos = match.start()
        
        try:
            obj, pos = decoder.raw_decode(document, pos)
        except ValueError:
            # do something sensible if there's some error
            raise ValueError("There was an issue when decoding stacked JSON.")
        yield obj

# Download and read stack json files from s3
def _download_stacked_json_from_s3(bucket_name, key_name, client):
    stacked_response_bytes = _download_from_s3(bucket_name, key_name, client)
    stacked_response_str = stacked_response_bytes.decode()
    response = _decode_stacked(stacked_response_str)
    return list(response)

# Method that returns the similarity between two strings
def _similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()