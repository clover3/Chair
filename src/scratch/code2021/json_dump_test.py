from cache import dump_to_json


def main():
    di = {
        'hi':'hello'
    }
    dump_to_json(di, 'json_dump_test')


if __name__ == "__main__":
    main()
