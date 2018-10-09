

from os.path import isfile
from os import access, R_OK, getcwd
import sys


def get_dict(filename):

    print(f"Current directory is '{getcwd()}'")

    assert isfile(filename), f"File not found: '{filename}'"
    assert access(filename, R_OK), f"File {filename} doesn't exist or isn't readable"

    f = open(filename)

    content = f.read().splitlines() 

    f.close()

    isFindComment = True

    keys = []
    queries = []

    while len (content) > 0:

        c = content.pop(0)

        if len(c) == 0:
            continue

        isComment = c.startswith("/*") and c.endswith("*/")

        isSQL = c.endswith(';')

        if isComment and isFindComment:
            c = c[2:]
            c = c[:-2]

            query_string = c.split(':')

            sql_key = query_string[0]

            keys.append(sql_key)

            isFindComment = False

        elif isSQL and not isFindComment:
            c = c[:-1]
            queries.append(c)
            isFindComment = True

        else:
            if isSQL:
                print(f"Skipped SQL line without prior comment: {c}")
            elif isComment:
                print(f"Skipped comment '{c}' because comment already given.")
            else:
                print(f"Skipped line '{c}'")
            
    if len(keys) > len (queries):
        keys.pop()

    assert len(keys) == len (queries)

    d = dict(zip(keys, queries))

    return d
  

"""c"""


if __name__ == "__main__":
    filename = "test_sql.txt"
    d = get_dict(filename)


for key, query in d.items():
    print(f"{key}: {query}")

