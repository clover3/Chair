import json
import sys


def process_jsonl_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            for key, value in json_obj.items():
                print(f"{key}: {value}")
            print()  # Print a blank line between JSON objects


def main():
    process_jsonl_file(sys.argv[1])


if __name__ == "__main__":
    main()

