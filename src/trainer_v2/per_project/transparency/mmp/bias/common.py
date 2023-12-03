from cpath import output_path
from list_lib import right
from misc_lib import path_join
from table_lib import tsv_iter


def find_indices(text_token, target_term):
    """
    Returns the indices of 'text_token' where one of the 'target_term' appears.

    Parameters:
    text_token (List[str]): The list of text tokens.
    target_term (List[str]): The list of target terms to search for.

    Returns:
    List[int]: A list of indices where any of the target terms appear in text_token.
    """
    indices = [i for i, token in enumerate(text_token) if token in target_term]
    return indices


def load_car_maker_list():
    term_list_path = path_join(output_path, "mmp", "bias", "car_maker_list.txt")
    term_list = [line.lower().strip() for line in open(term_list_path, "r")]
    return term_list


def load_second_column_from_bias_dir(file_name):
    itr = tsv_iter(path_join(output_path, "mmp", "bias", file_name))
    return right(list(itr))


def load_from_bias_dir(file_name):
    term_list_path = path_join(output_path, "mmp", "bias", file_name)
    term_list = [line.lower().strip() for line in open(term_list_path, "r")]
    return term_list


def contain_any(text, target_terms):
    tokens = text.lower().split()
    lower = text.lower()
    for term in target_terms:
        if term in lower:
            return True

    return False