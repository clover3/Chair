from crs.load_stance_annotation import load_stance_verify_annot
from collections import Counter




def agreement(path):
    data = load_stance_verify_annot(path)
    return agreement_inner(data)


def agreement_per_type(path):

    data = load_stance_verify_annot(path)

    article_data = []
    comment_data = []

    def get_seg_id(link):
        seg_id = link.split("/")[-2]
        return int(seg_id)

    for e in data:
        if get_seg_id(e['link']) < 100:
            article_data.append(e)
        else:
            comment_data.append(e)

    print(len(article_data), "Article")
    agreement_inner(article_data)

    print(len(comment_data), "comments")
    agreement_inner(comment_data)


def agreement_inner(data):
    group = {}
    for e in data:
        sig = e['statement'] + e['link']
        if sig not in group:
            group[sig] = []

        group[sig].append((e['support'], e['dispute']))

    order1 = 0
    order2 = 1

    s_count1 = Counter()
    s_count2 = Counter()
    d_count1 = Counter()
    d_count2 = Counter()
    s_agree = 0
    d_agree = 0
    for sig in group:
        #print(statement)
        support1, dispute1 = group[sig][order1]
        support2, dispute2 = group[sig][order2]
        s_count1[support1] += 1
        s_count2[support2] += 1
        d_count1[dispute1] += 1
        d_count2[dispute2] += 1

        if support1 == support2:
            s_agree += 1
        if dispute1 == dispute2:
            d_agree += 1
        #print(group[statement])

    def sq(x):
        return x*x

    total = len(group) * 2
    s_po = s_agree / total
    d_po = d_agree / total


    s_pe = 0
    d_pe = 0
    for i in range(3):
        s_pe += (s_count1[i] / total ) * (s_count2[i] / total)
        d_pe += (d_count1[i] / total) * (d_count2[i] / total)


    print("Support p_0 : ", s_po)
    print("Dispute p_0 : ", d_po)

    s_kappa = (s_po - s_pe) / (1 - s_pe)
    d_kappa = (d_po - d_pe) / (1 - d_pe)

    print("Support kappa : ", s_kappa)
    print("Dispute kappa : ", d_kappa)



def combine(path):
    data = load_stance_verify_annot(path)

    group = {}
    for e in data:
        sig = e['statement'] + e['link']
        if sig not in group:
            group[sig] = []

        group[sig].append((e['support'], e['dispute']))

    order1 = 0
    order2 = 1

    s_count = Counter()
    d_count = Counter()
    s_agree = 0
    d_agree = 0

    def unanimous(l):
        if len(l) < 2:
            return False

        for item in l:
            if item != l[0]:
                return False
        print(l)
        return True

    NOT_FOUND = 0
    YES = 1
    NOT_SURE = 2

    result = {}
    for statement in group:
        s_list = []
        d_list = []
        for s, d in group[statement]:
            if s != NOT_SURE:
                s_list.append(s)
            if d != NOT_SURE:
                d_list.append(d)

        if unanimous(s_list):
            if s_list[0] == 1 :
                s, loc = statement.split("http://gosford.cs.umass.edu/search/#!/view3/")
                if s not in result:
                    result[s] = []
                result[s].append(('support', loc))

        if unanimous(d_list):
            if d_list[0] == 1:
                s, loc = statement.split("http://gosford.cs.umass.edu/search/#!/view3/")
                if s not in result:
                    result[s] = []
                result[s].append(('dispute', loc))

    for s in result:
        print(s)
        print(result[s])

def summary(path):
    data = load_stance_verify_annot(path)

    group = {}
    sig2data = {}
    for e in data:
        sig = e['statement'] + e['link']
        sig2data[sig] = e['statement'], e['link']
        if sig not in group:
            group[sig] = []

        group[sig].append((e['support'], e['dispute']))

    NOT_FOUND = 0
    YES = 1
    NOT_SURE = 2

    for sig in group:
        statement, link = sig2data[sig]
        print("------------")
        print("Statement: ", statement)
        print("Text : ", link)
        cnt = 1
        for s, d in group[sig]:
            s_word = {
                NOT_FOUND: "NotSupport",
                YES: "Support",
                NOT_SURE:"NotSure",
            }[s]

            d_word = {
                NOT_FOUND: "NotDispute",
                YES: "Dispute",
                NOT_SURE: "NotSure",
            }[d]
            print("CrowdWorker#{}: ".format(cnt), s_word, d_word)
            cnt += 1


if __name__ == "__main__":
    path = "C:\work\Data\CKB annotation\\verify stance -1\\Batch_3784489_batch_results.csv"
    agreement(path)



