import json


def parse_comments(file_path):
    d  = open(file_path, "rb").read()
    d = d[7:-3]
    #f = open(file_path, encoding='utf-8', errors='ignore')
    j = json.loads(d)
    info = {
        'comments' : []
    }
    try:
        discussion = j['discussion']
        current_page = j['currentPage']
        discussion_id = discussion['key']
        info['short_id'] = discussion_id
        info['webUrl'] = discussion['webUrl']
        comment_count = j['discussion']['commentCount']
        comments = j['discussion']['comments']
        for comment in comments:
            comment_id = str(comment['id'])
            comment_body = comment['body']
            head = comment_id, comment_body
            tail = []
            if 'responses' in comment:
                for response in comment['responses']:
                    res_body = response['body']
                    res_id = str(response['id'])
                    comment_target = response['responseTo']['commentId']
                    tail.append((res_id, res_body, comment_target))
            info['comments'].append((head, tail))
    except KeyError as e:
        print(file_path)
    return info