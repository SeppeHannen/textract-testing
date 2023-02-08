from src.utils import _similarity
from src.state import _get_documents_with_truth, _get_query_to_documents, _analyze_document

# Call Amazon Textract's Analyze Document using `query_hypothesis` as the query on all documents that
# the Ground Truth labeling workforce has specified a Ground Truth for the query `query`.
# TODO: Speed this method up by making it asynchronous.
def _get_textract_responses(query, truth):
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
        textract_response = _analyze_document(document, query, truth)
        textract_responses[document] = textract_response['Blocks'][-1]['Text']

    return textract_responses


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

    