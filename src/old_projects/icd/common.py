from old_projects.icd.input_path_getter import get_icm10cm_order_path


def lmap(func, iterable_something):
    return list([func(e) for e in iterable_something])


def load_description():
    path = get_icm10cm_order_path()
    f = open(path, "r")

    def parse_line(line):
        order_number = line[:5]
        assert line[5:6] == " "
        icd10_code = line[6:13]
        assert line[13:14] == " "
        is_header = line[14:15]
        assert line[15:16] == " "
        short_desc = line[16:76]
        assert line[76:77] == " "
        long_desc = line[77:]
        return {
            'order_number': int(order_number),
            'icd10_code': icd10_code,
            'is_header': is_header,
            'short_desc': short_desc.strip(),
            'long_desc': long_desc.strip(),
        }

    data = lmap(parse_line, f.readlines())
    return data


def AP_from_binary(is_correct_list, num_gold):
    tp = 0
    sum = 0
    for idx, is_correct in enumerate(is_correct_list):
        n_pred_pos = idx + 1
        if is_correct:
            tp += 1
            prec = (tp / n_pred_pos)
            assert prec <= 1
            sum += prec
    assert sum <= num_gold
    return sum / num_gold