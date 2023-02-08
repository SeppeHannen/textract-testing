import boto3
import json
import re
from json import JSONDecoder, JSONDecodeError
from difflib import SequenceMatcher

# Define global variables used to keep track of the state of the Ground Truth
global query_to_document
global document_to_truth

# Create AWS clients for required services
s3_client = boto3.client('s3')
sm_client = boto3.client('sagemaker')
textract_client = boto3.client('textract')

# Method that returns the similarity between two strings
def _similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Download an object from S3 as bytes
def _download_from_s3(bucket_name, key_name):
    response = s3_client.get_object(Bucket=bucket_name, Key=key_name)['Body']
    return response.read()

# Download an object from S3 as JSON
def _download_json_from_s3(bucket_name, key_name):
    response = _download_from_s3(bucket_name, key_name)
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
        except JSONDecodeError:
            # do something sensible if there's some error
            raise
        yield obj

# Download and read stack json files from s3
def _download_stacked_json_from_s3(bucket_name, key_name):
    stacked_response_bytes = _download_from_s3(bucket_name, key_name)
    stacked_response_str = stacked_response_bytes.decode()
    response = _decode_stacked(stacked_response_str)
    return list(response)

# Get a map of all documents which contain a truth for the input query mapped to those truths
def _get_documents_with_truth(query):
    global document_to_truth
    documents = query_to_document[query]
    return {document: document_to_truth[document][query] for document in documents}

# Get the S3 Uris of all documents that contain a truth for the input query
def _get_query_to_documents(query):
    global query_to_document
    return query_to_document[query]

# Call Amazon Textract's Analyze Document using `query_hypothesis` as the query on all documents that
# the Ground Truth labeling workforce has specified a Ground Truth for the query `query`.
# TODO: Speed this method up by making it asynchronous.
def _get_textract_responses(query, query_hypothesis):
    documents = _get_query_to_documents(query)
    
    if not len(documents) > 0:
        raise ValueError(
            """
            There are no documents containing a truth for this query in the Ground Truth.
            Consider calling `update_ground_truth()` to update this libraries state of the Truth.
            If that does not help make sure that your Sagemaker Ground Truth labeling job
            contains labels for the query you were trying to test.
            """
            )

    textract_responses = {}
    for document in documents:
        textract_response = textract_client.analyze_document(
            Document={
                'S3Object': {
                    'Bucket': document.split('/', 3)[2],
                    'Name': document.split('/', 3)[3]
                }
            },
            FeatureTypes=['QUERIES'],
            QueriesConfig={
                'Queries': [
                    {
                        'Text': query_hypothesis,
                        'Alias': query,
                    },
                ]
            }
        )
        textract_responses[document] = textract_response['Blocks'][-1]['Text']

    return textract_responses

# Updates the global variables which are used to compute the accuracy (`query_to_document`, `document_to_truth`)
# Make sure to specify your labeling_job_name, you can find this in the Sagemaker Ground Truth Labeling jobs section
def update_ground_truth(labeling_job_name):
    global document_to_truth, query_to_document

    # Make API call to Sagemaker to get the output S3 Uri of the Ground Truth labeling job.
    labeling_job_response = sm_client.describe_labeling_job(
        LabelingJobName=labeling_job_name)
    labeling_job_output_path = labeling_job_response['OutputConfig']['S3OutputPath']

    # Get the output manifest file
    labeling_job_output_manifest_path = labeling_job_output_path + labeling_job_name + '/manifests/output/output.manifest'

    # Download the manifest file from S3
    manifest_json = _download_stacked_json_from_s3(
        bucket_name=labeling_job_output_path.split('/')[2],
        key_name=labeling_job_output_manifest_path.split('/', 3)[3],
    )

    # Remove the existing state of the Ground Truth
    query_to_document = {}
    document_to_truth = {}

    # For every query get the documents that have a label for that query
    for document in manifest_json:
        # Add the document to the keys of the mapping for answers
        document_to_truth[document['source-ref']] = {}

        for query in document['metadata']['labels']:
            # Make sure to add all queries to the `query_to_document` map
            if query not in query_to_document:
                query_to_document.update({query: []})

            # Add the document to the query in the map
            query_to_document[query].append(document['source-ref'])

            # Download the truth of the document
            truth_response_json = _download_json_from_s3(
                bucket_name=document[labeling_job_name]['annotation-ref'].split('/')[2],
                key_name=document[labeling_job_name]['annotation-ref'].split('/', 3)[3]
            )
            truth_entities = truth_response_json['Entities']
            truth = next((entity['Text'] for entity in truth_entities if entity['Type'] == query), None)

            # Add the truth of this query to the `document_to_truth` map
            document_to_truth[document['source-ref']][query] = truth


class Hypothesis:
    query: str
    truth: str
    # Contains the responses of the API calls made to Amazon Textract
    textract_responses: list

    def __init__(self, query):
        self.query = query

    def add_truth(self, query):
        self.truth = query

    def get_accuracy(self):
        # Make calls to Amazon Textract.
        if self.query is None or self.truth is None:
            raise ValueError(
                """
                The query and the corresponding truth need to be defined to be able to make calls 
                to Amazon Textract.
                """
                )
        self.textract_responses = _get_textract_responses(self.query, self.truth)

        # Get all relevant documents and Ground Truths for this query
        query_truth_documents = _get_documents_with_truth(self.query)

        # TODO: Speed this up with vectorization
        sum = 0

        results = []
        # Compute the similarity between the Ground Truths and the Amazon Textract responses
        for document in self.textract_responses.keys():
            sum += _similarity(self.textract_responses[document], query_truth_documents[document])
            results.append(
                    {
                        'Document': document, 
                        'Textract': self.textract_responses[document], 
                        'Truth': query_truth_documents[document]
                    }
                )
        accuracy = sum/len(self.textract_responses.keys())

        return accuracy, results

    