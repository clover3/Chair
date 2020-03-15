import pickle

import pymysql

pymysql.install_as_MySQLdb()

import dataset

from datastore.sql_path import sydney_mysql_path

db = None


def get_sydney_database():
    return dataset.connect(sydney_mysql_path)


def check_init_db():
    global db
    if db is None:
        db = get_sydney_database()


def save(table_name, key, value):
    check_init_db()
    byte_arr = pickle.dumps(value)
    table = db[table_name]
    table.insert(dict(key=key, value=byte_arr))


def load(table_name, key):
    check_init_db()
    table = db[table_name]
    entry = table.find_one(key=key)
    if entry is None:
        raise KeyError()
    byte_arr = entry['value']
    return pickle.loads(byte_arr)


def basic_test():
    value = [1,2,312941,2291]
    save("test_table", "doc_ab", value)
    print(load("test_table", "doc_ab"))


def dictionary_test(table_name, key):
    print(load(table_name, key))


if __name__ == "__main__":
    basic_test()
    #dictionary_test(sys.argv[0], sys.argv[1])