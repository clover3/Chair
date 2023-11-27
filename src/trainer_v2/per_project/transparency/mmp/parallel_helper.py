from multiprocessing import Pool


def parallel_run(input_list, list_fn, split_n):
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            print(i, i + n)
            yield l[i:i + n]

    p = Pool(split_n)

    item_per_job = (len(input_list) + split_n - 1) // split_n
    print("item_per_job", item_per_job)
    l_args = chunks(input_list, item_per_job)

    result_list_list = p.map(list_fn, l_args)

    result = []
    for result_list in result_list_list:
        result.extend(result_list)
    return result