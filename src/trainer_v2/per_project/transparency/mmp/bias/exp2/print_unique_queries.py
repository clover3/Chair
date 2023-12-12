import sys

from iter_util import load_jsonl


def main():
    file_path = sys.argv[1]
    relevant_entries = load_jsonl(file_path)

    queries = set()
    for item in relevant_entries:
        queries.add(item["query"])

    for item in queries:
        print(item)


if __name__ == "__main__":
    main()
