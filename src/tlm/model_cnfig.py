import copy
import json

import six

from data_generator.tf_gfile_support import tf_gfile


class JsonConfig(object):
    """Configuration for `BertModel`."""
    def __init__(self):
        pass

    @classmethod
    def from_dict(cls, json_object):
        config = JsonConfig()
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with tf_gfile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def dummy(self):
        NotImplemented

    def compare_attrib_value_safe(self, attrib, value):
        if hasattr(self, attrib):
            return self.__dict__[attrib] == value
        return False
