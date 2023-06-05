import gzip
import json
import sys


class CustomEncoder(json.JSONEncoder):
    def iterencode(self, o, _one_shot=False):
        if isinstance(o, float):
            yield format(o, '.4g')
        elif isinstance(o, list):
            yield '['
            first = True
            for value in o:
                if first:
                    first = False
                else:
                    yield ', '
                yield from self.iterencode(value)
            yield ']'
        else:
            yield from super().iterencode(o, _one_shot=_one_shot)


def dev1():
    data = {
        "value": 123.456789,
        "array": [1.23456789, 2.3456789, 3.456789]
    }
    data = [123.456789, [1.23456789, 2.3456789, 3.456789], 0.1234, 1]

    json_string = json.dumps(data, cls=CustomEncoder)
    print(json_string)

    recovered = json.loads(json_string)
    print(recovered)


def main():
    f = open(sys.argv[1], "r")
    f_out = gzip.open(sys.argv[2], 'wt', encoding='utf8')
    for line in f:
        j = json.loads(line)
        s = json.dumps(j, cls=CustomEncoder)
        f_out.write(s + "\n")



if __name__ == "__main__":
    main()