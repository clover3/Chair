import inspect
import re


def log_variables(*var_list):
    lines = inspect.getframeinfo(inspect.currentframe().f_back)[3]
    for line in lines:
        m = re.search(r'log_variables\s*\(\s*([\s,A-Za-z0-9_]*)\s*\)', line)
        if m:
            raw_var_string = m.group(1)
            var_strings = raw_var_string.split(",")
            for var_name, var in zip(var_strings, var_list):
                print(var_name.strip(), var)
        else:
            print("log variables failed")
