

import boto3

from contradiction.medical_claims.mturk.mturk_api_common import get_client
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


def make_qualification_w_no(no):
    questions = open(at_data_dir("mturk", "password_question.txt"), mode='r').read()
    answers = open(at_data_dir("mturk", "password_answer{}.txt".format(no)), mode='r').read()
    sandbox_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
    active_url = 'https://mturk-requester.us-east-1.amazonaws.com'
    mturk = boto3.client('mturk',
                          region_name='us-east-1',
                          endpoint_url=sandbox_url)

    qual_response = mturk.create_qualification_type(
                            Name='Password Authentication{}'.format(no),
                            Keywords='Password Authentication{}'.format(no),
                            RetryDelayInSeconds=100,
                            Description='Password Authentication{}'.format(no),
                            QualificationTypeStatus='Active',
                            Test=questions,
                            AnswerKey=answers,
                            TestDurationInSeconds=300)

    print(qual_response['QualificationType']['QualificationTypeId'])



def update_qualification():
    questions = open(at_data_dir("mturk", "question_0.txt"), mode='r').read()
    answers = open(at_data_dir("mturk", "answer_0_score.txt"), mode='r').read()
    mturk = get_client()
    type_id = "36N1A6SSIK1W1ZSKTY6LMC0K0C3BTU"
    qual_response = mturk.update_qualification_type(
        QualificationTypeId=type_id,
        RetryDelayInSeconds=10000 * 10000,
        Description='Closed',
        Test=questions,
        AnswerKey=answers,
        TestDurationInSeconds=30)
    print(qual_response)
    print(qual_response['QualificationType']['QualificationTypeId'])


def revoke_qualifications():
    type_id = "3PO9K4KN9685G9OOQNH4PR63WORY7H"
    active_url = 'https://mturk-requester.us-east-1.amazonaws.com'
    mturk = boto3.client('mturk',
                     region_name='us-east-1',
                     endpoint_url=active_url)
    next_token = ""
    print(mturk.__class__.__dict__.keys())

    while True:
        if next_token:
            res = mturk.list_workers_with_qualification_type(QualificationTypeId=type_id,
                                                             MaxResults=100,
                                                             NextToken=next_token)
        else:
            res = mturk.list_workers_with_qualification_type(QualificationTypeId=type_id,
                                                             MaxResults=100,
                                                             )

        print("{} qualifications found".format(len(res['Qualifications'])))
        for d in res['Qualifications']:
            worker_id = d["WorkerId"]
            ret = mturk.disassociate_qualification_from_worker(WorkerId=worker_id, QualificationTypeId=type_id)
            print(ret)
        if 'NextToken' in res:
            next_token = res['NextToken']
        else:
            break


def main():
    revoke_qualifications()


if __name__ == "__main__":
    main()
