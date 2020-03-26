import pickle
import random

from sqlalchemy.orm.exc import MultipleResultsFound, NoResultFound

from datastore.alchemy_schema import Session, table_class_by_name

session = None


def get_sydney_database():
    return Session()


def check_init_db():
    global session
    if session is None:
        session = get_sydney_database()


def save(table_name, key, value):
    check_init_db()
    table_class = table_class_by_name[table_name]
    byte_arr = pickle.dumps(value)

    new_record = table_class(key=key, value=byte_arr)
    session.add(new_record)
    session.flush()


def save_wo_flush(table_name, key, value):
    check_init_db()
    table_class = table_class_by_name[table_name]
    byte_arr = pickle.dumps(value)
    if len(byte_arr) > 500 * 1024 * 1024:
        print("Skip large data : ", len(byte_arr))
        return

    new_record = table_class(key=key, value=byte_arr)
    session.add(new_record)


def bulk_save(table_name, key_and_value_list):
    check_init_db()
    table_class = table_class_by_name[table_name]
    save_payload = []
    for key, value in key_and_value_list:
        byte_arr = pickle.dumps(value)
        if len(byte_arr) > 500 * 1024 * 1024:
            print("Skip large data : ", len(byte_arr))
            return
        new_record = table_class(key=key, value=byte_arr)
        save_payload.append(new_record)

    session.bulk_save_objects(save_payload)
    session.commit()


def flush():
    if session is not None:
        session.flush()
        session.commit()


def load(table_name, key):
    check_init_db()
    preload_check = preload_man.lookup(table_name, key)
    if preload_check == CONTAIN:
        return preload_man.get(table_name, key)
    elif preload_check == NOT_IN_DB:
        raise KeyError()
    elif preload_check == NOT_FOUND:
        pass
    else:
        assert False
    table_class = table_class_by_name[table_name]
    try:
        q_res = session.query(table_class).filter(table_class.key == key).one()
        byte_arr = q_res.value
        return pickle.loads(byte_arr)
    except MultipleResultsFound as e:
        raise KeyError()
    except NoResultFound as e:
        raise KeyError()


def load_all(table_name):
    table_class = table_class_by_name[table_name]
    q_res_itr = session.query(table_class).all()
    for row in q_res_itr:
        byte_arr = row.value
        yield pickle.loads(byte_arr)


def load_multiple(table_name, keys, unpickle=False):
    check_init_db()
    table_class = table_class_by_name[table_name]
    out_d = {}
    try:
        q_res = session.query(table_class).filter(table_class.key.in_(keys)).all()
        for row in q_res:
            if unpickle:
                out_d[row.key] = pickle.loads(row.value)
            else:
                out_d[row.key] = row.value
    except MultipleResultsFound as e:
        raise KeyError()
    except NoResultFound as e:
        raise KeyError()
    return out_d


def get_existing_keys(table_name, keys):
    check_init_db()
    table_class = table_class_by_name[table_name]
    try:
        q_res = session.query(table_class.key).filter(table_class.key.in_(keys)).all()
        out = list([row.key for row in q_res])

    except MultipleResultsFound as e:
        raise KeyError()
    except NoResultFound as e:
        raise KeyError()
    return out


def has_key(table_name, key):
    check_init_db()
    table_class = table_class_by_name[table_name]
    try:
        q_res = session.query(table_class).filter(table_class.key == key).one()
        byte_arr = q_res.value
        return True
    except NoResultFound as e:
        return False


def basic_test():
    value = [1,2,312941,2291]
    ## nono
    num = random.randint(1, 1000)
    key ="doc_ab_{}".format(num)
    save("test_table", key, value)
    print(load("test_table", key))


def dictionary_test(table_name, key):
    print(load(table_name, key))


def get_sliced_rows(table_name, start, end):
    check_init_db()

    table_class = table_class_by_name[table_name]
    try:
        q_res = session.query(table_class).order_by(table_class.key).offset(start).limit(end-start)
        for row in q_res:
            yield row.key, pickle.loads(row.value)
    except MultipleResultsFound as e:
        raise KeyError()
    except NoResultFound as e:
        raise KeyError()


def sliced_row_test():
    for key, value in get_sliced_rows("TokenizedCluewebDoc", 10, 20):
        print(key)


CONTAIN = 2
NOT_IN_DB = 1
NOT_FOUND = 0


class PreloadMan:
    def __init__(self):
        self.data_d = {}
        self.not_found_d = {}

    def preload(self, table_name, keys):
        found_d = load_multiple(table_name, keys)
        not_found_list = [k for k in keys if k not in found_d]

        if table_name not in self.data_d:
            self.data_d[table_name] = {}
            self.not_found_d[table_name] = set()

        for key, value in found_d.items():
            self.data_d[table_name][key] = pickle.loads(value)

        self.not_found_d[table_name].update(not_found_list)

    def do_empty(self):
        self.data_d = {}
        self.not_found_d = {}

    def lookup(self, table_name, key):
        if table_name in self.data_d and key in self.data_d[table_name]:
            return CONTAIN
        elif table_name in self.data_d and key in self.not_found_d[table_name]:
            return NOT_IN_DB
        else:
            return NOT_FOUND

    def get(self, table_name, key):
        return self.data_d[table_name][key]


preload_man = PreloadMan()

if __name__ == "__main__":
    sliced_row_test()
    #dictionary_test(sys.argv[0], sys.argv[1])
