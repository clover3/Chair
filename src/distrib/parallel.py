from multiprocessing import Pool


def parallel_run(input_list, common_arg, list_fn, split_n):
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    p = Pool(split_n)
    l_args = chunks(input_list, split_n)

    args = [(a, common_arg) for a in l_args]
    result_list_list = p.map(list_fn, args)

    result = []
    for result_list in result_list_list:
        result.extend(result_list)
    return result