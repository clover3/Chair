import copy
import json
import os


class JsonTiedList:
    def __init__(self, file_path):
        self.file_path = file_path
        if os.path.exists(file_path):
            self.list = json.load(open(self.file_path, "r"))
        else:
            self.list = []

    def save(self):
        with open(self.file_path, "w") as f:
            json.dump(self.list, f)

    def add(self, value):
        self.list.append(value)
        self.save()

    def remove(self, value):
        self.list.remove(value)
        self.save()

    def __iter__(self):
        for value in self.list:
            yield value


class JsonTiedDict:
    def __init__(self, file_path):
        self.file_path = file_path
        if os.path.exists(file_path):
            dict = json.load(open(self.file_path, "r"))

        else:
            dict = {
                "last_task_id": 0
            }
        for key, value in dict.items():
            self.__dict__[key] = value

    def last_id(self):
        return self.__dict__['last_task_id']

    def save(self):
        with open(self.file_path, "w") as f:
            json.dump(self.to_dict(), f)

    def set(self, key, value):
        self.__dict__[key] = value
        self.save()

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output