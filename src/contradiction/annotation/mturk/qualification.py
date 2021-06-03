

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


def make_hit():
    '367ZSS7NPQL6UUVQ8416MQCQ3408AT'
    qualification_type_id = '32R8QD8BQAA644OF9Q8DEDGFEYIDC6'
    hit = client.create_hit(
            Reward='0.01',
            LifetimeInSeconds=3600,
            AssignmentDurationInSeconds=600,
            MaxAssignments=9,
            Title='A HIT with a qualification test',
            Description='A test HIT that requires a certain score from a qualification test to accept.',
            Keywords='boto, qualification, test',
            AutoApprovalDelayInSeconds=0,
            QualificationRequirements=[{'QualificationTypeId':'3CFGE88WF7UDUETM7YP3RSRD73VS4F',
                                       'Comparator': 'EqualTo',
                                       'IntegerValues':[100]}]
            )

make_qualification()