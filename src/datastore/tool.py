from datastore.interface import save, flush, has_key, save_wo_flush, get_existing_keys, bulk_save


class BufferedSaver:
    def __init__(self):
        self.buffer = []

    def save(self, table_name, key, value):
        e = table_name, key, value
        self.buffer.append(e)
        if len(self.buffer) > 1000:
            self.flush()

    def flush(self):
        pass
        for table_name, key, value in self.buffer:
            save(table_name, key, value)
        flush()
        self.buffer = []


class PayloadSaver:
    def __init__(self):
        self.buffer = []

    def save(self, table_name, key, value):
        e = table_name, key, value
        self.buffer.append(e)


def commit_buffer_to_db(buffer):
    cnt = 0
    skipped_keys = 0
    for e in buffer:
        table_name, key, value = e
        if has_key(table_name, key):
            skipped_keys += 1
            pass
        else:
            if skipped_keys:
                print("Skipped {} existing keys".format(skipped_keys))
                skipped_keys = 0
            cnt += 1
            save(table_name, key, value)

        if cnt > 100:
            flush()


def commit_buffer_to_db_batch(buffer):
    buffer.sort(key=lambda x:x[0])
    def block_iterator():
        last_table_name = ""
        key_values = []
        for e in buffer:
            table_name, key, value = e
            if table_name == last_table_name:
                key_values.append((key, value))
            else:
                if key_values:
                    yield last_table_name, key_values
                key_values = [(key, value)]
                last_table_name = table_name

        if key_values:
            yield last_table_name, key_values

    for table_name, block in block_iterator():
        keys, values= zip(*block)
        exist_keys = get_existing_keys(table_name, keys)
        exist_keys = set(exist_keys)
        key_values_to_insert = [(key, value) for key , value in block if key not in exist_keys]

        if exist_keys:
            print("Skipped {} existing keys".format(len(exist_keys)))
        bulk_save(table_name, key_values_to_insert)

        flush()


def commit_buffer_to_db2(buffer):
    cnt = 0
    print(buffer[0][0])
    for e in buffer:
        table_name, key, value = e
        try:
            cnt += 1

            save_wo_flush(table_name, key, value)

            if cnt > 100:
                print("Flush")
                flush()
        except Exception as e:
            print(e)
            pass

