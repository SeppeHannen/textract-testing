from textract_testing import Hypothesis, update_ground_truth

update_ground_truth('textract-pdf-labeling-job-1-labeling-job-20230202T120436')

query_1 = Hypothesis('What is the name of the certification company?')
query_1.add_truth('What is the name of the certification company?')

print(query_1.get_accuracy())