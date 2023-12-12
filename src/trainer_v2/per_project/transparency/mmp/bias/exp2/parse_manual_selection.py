
import sys

def parse_dict_on_lines(file_path):
    text = open(file_path, "r").read()
    # Split the text into items by empty lines
    items = text.strip().split('\n\n')

    list_of_dicts = []
    for item in items:
        # Split each item into lines
        lines = item.split('\n')
        item_dict = {}
        for line in lines:
            if line.strip():
                # Split each line into key-value pairs
                key, value = line.split(': ', 1)
                item_dict[key.strip()] = value.strip()
        list_of_dicts.append(item_dict)

    return list_of_dicts
