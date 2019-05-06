import re

def line_split(line):
  s1 = re.sub(r'(\w[A-Z]|[0-9a-z])([.!?]) ([A-Z])', r'\1\2__|__\3', line)
  return s1.split('__|__')
