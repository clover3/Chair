


import boto3

from cpath import at_data_dir


def make_qualification():
    desc = "This test will show you the instruction and ask to answer appropriate category for each documents."
    questions = open(at_data_dir("mturk", "question_ca.txt"), mode='r').read()
    answers = open(at_data_dir("mturk", "answer_ca.txt"), mode='r').read()
    # sandbox_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
    active_url = 'https://mturk-requester.us-east-1.amazonaws.com'
    mturk = boto3.client('mturk',
                          region_name='us-east-1',
                          endpoint_url=active_url)

    qual_response = mturk.create_qualification_type(
                            Name='CA-test',
                            Keywords='CA-test',
                            RetryDelayInSeconds=1,
                            Description=desc,
                            QualificationTypeStatus='Active',
                            Test=questions,
                            AnswerKey=answers,
                            TestDurationInSeconds=3600)

    print(qual_response['QualificationType']['QualificationTypeId'])


make_qualification()