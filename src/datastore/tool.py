from datastore.interface import save, flush, has_key


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
    for e in buffer:
        table_name, key, value = e
        if has_key(table_name, key):
            pass
        else:
            save(table_name, key, value)
    flush()

