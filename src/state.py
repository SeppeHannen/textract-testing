import boto3
from src.utils import _download_json_from_s3, _download_stacked_json_from_s3

# Define global variables used to keep track of the state of the Ground Truth
global query_to_document
global document_to_truth

# Create AWS clients for required services
s3_client = boto3.client('s3')
sm_client = boto3.client('sagemaker')
textract_client = boto3.client('textract')

# Get a map of all documents which contain a truth for the input query mapped to those truths
def _get_documents_with_truth(query):
    global document_to_truth
    documents = query_to_document[query]
    return {document: document_to_truth[document][query] for document in documents}

# Get the S3 Uris of all documents that contain a truth for the input query
def _get_query_to_documents(query):
    global query_to_document
    return query_to_document[query]

# Make an API call to Amazon Textract to analyze a document using a query
def _analyze_document(document, query, truth):
    return textract_client.analyze_document(
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
                    'Text': truth,
                    'Alias': query,
                },
            ]
        }
    )

# Return a list of all queries that have a ground truth available
def get_truths():
    global query_to_document
    return query_to_document.keys()

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
        client=s3_client
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
                key_name=document[labeling_job_name]['annotation-ref'].split('/', 3)[3],
                client=s3_client
            )
            truth_entities = truth_response_json['Entities']
            truth = next((entity['Text'] for entity in truth_entities if entity['Type'] == query), None)

            # Add the truth of this query to the `document_to_truth` map
            document_to_truth[document['source-ref']][query] = truth