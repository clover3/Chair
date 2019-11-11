from tlm.retrieve_lm.mysql_sentence import get_doc_sent





if __name__ == "__main__":
    rows = get_doc_sent('FT931-7027')
    print(type(rows))