

import boto3

from cpath import at_data_dir


def make_qualification():
    questions = open(at_data_dir("mturk", "password_question.txt"), mode='r').read()
    answers = open(at_data_dir("mturk", "password_answer.txt"), mode='r').read()
    sandbox_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
    active_url = 'https://mturk-requester.us-east-1.amazonaws.com'
    mturk = boto3.client('mturk',
                          region_name='us-east-1',
                          endpoint_url=active_url)

    qual_response = mturk.create_qualification_type(
                            Name='Password Authentication',
                            Keywords='Password Authentication',
                            RetryDelayInSeconds=1,
                            Description='Password Authentication',
                            QualificationTypeStatus='Active',
                            Test=questions,
                            AnswerKey=answers,
                            TestDurationInSeconds=300)

    print(qual_response['QualificationType']['QualificationTypeId'])


def make_qualification2():
    questions = open(at_data_dir("mturk", "password_question.txt"), mode='r').read()
    answers = open(at_data_dir("mturk", "password_answer2.txt"), mode='r').read()
    sandbox_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
    active_url = 'https://mturk-requester.us-east-1.amazonaws.com'
    mturk = boto3.client('mturk',
                          region_name='us-east-1',
                          endpoint_url=active_url)

    qual_response = mturk.create_qualification_type(
                            Name='Password Authentication2',
                            Keywords='Password Authentication2',
                            RetryDelayInSeconds=1,
                            Description='Password Authentication2',
                            QualificationTypeStatus='Active',
                            Test=questions,
                            AnswerKey=answers,
                            TestDurationInSeconds=300)

    print(qual_response['QualificationType']['QualificationTypeId'])


make_qualification2()