import pandas as pd
import numpy as np
import json
d = ['','T','M','U']
def parse(row):
    name = row[3].strip()
    t = str(row[0])[:7],[str(row[1]), name,d[row[2]]]
    return t

if __name__ == '__main__':
    data = pd.read_csv("json.csv", names=["a","b","c","d"],encoding='utf-8').fillna('')
    n = [parse(l) for l in data.values]
    m = {k:v for k,v in n}
    print(m)
    with open("hcode.json", "w") as f:
        json.dump(m, f)