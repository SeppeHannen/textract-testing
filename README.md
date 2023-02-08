# textract-testing

Steps:
1. Follow [this](https://docs.aws.amazon.com/comprehend/latest/dg/cer-annotation-pdf.html#cer-annotation-pdf-set-up) guide for creating a labeling job for PDFs using Sagemaker Ground Truth.
2. Open the AWS Management Console and navigate to the labeling jobs section of Amazon Sagemaker Ground Truth. Find the labeling job you created by executing the script and make sure the status is "in progress".
3. Open your email (assuming you added yourself to the Private Labeling Workforce) and start labeling your PDFs. Once done you can use the the `test.ipynb` file to see whether it worked.

Note: 
- In the "Creating an annotation job" phase you might run in to issues when copy pasting the command (which invokes the script) into your terminal, if you do you may want to create a small shell script containing the commands and execute that instead.
- In the "Creating an annotation job" phase make sure that for every different part of the document that you would like to extract, that you create a different 'entity-type' that corresponds to it. If you want to extract the name of the certification company you could call define one entity as "the name of the certification company" or as "what is the name of the certification company?". A clear naming convention will help your labelers understand how to label the documents.
- If the status of your labeling job is "failed" then open a terminal and use the command `aws sagemaker describe-labeling-job --labeling-job-name <job_name>` to see more details regarding why it failed.