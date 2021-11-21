from tlm.qtype.runner.spacy_parse_queires import run_query_parse_jobs


def main():
    run_query_parse_jobs("dev")
    run_query_parse_jobs("test")


if __name__ == "__main__":
    main()