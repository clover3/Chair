from itertools import islice

src_itr = range(1000)


for job_no in range(10, 20):

    st = job_no * 10
    ed = st + 10
    itr = islice(src_itr, st, ed)
    print(job_no, list(itr))