



import boto3

from cpath import at_data_dir


def view_results():

    questions = open(at_data_dir("mturk", "password_question.txt"), mode='r').read()
    answers = open(at_data_dir("mturk", "password_answer.txt"), mode='r').read()
    sandbox_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
    active_url = 'https://mturk-requester.us-east-1.amazonaws.com'
    mturk = boto3.client('mturk',
                         region_name='us-east-1',
                         endpoint_url=active_url)

    hit_id = ""
    qual_response = mturk.get_assignments(hit_id)

