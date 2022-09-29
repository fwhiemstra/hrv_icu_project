import pandas as pd
import numpy as np

z = [1, 10, 100, 'z']
y = [2, 20, 200, 'y']
x = [3, 30, 300, 'x']

lst = []
lst.append(z)
lst.append(y)
lst.append(x)
print(lst)

keys = ['single', 'tens', 'hundreds', 'letters']

df = pd.DataFrame(lst, columns=keys)

print(df)

try:
        tuplekeys_all = batch_all[0].keys()
        lst_all = []

        for i in range(1,len(batch_all)):
            lst = []
            for key in tuplekeys_all:
                lst += [batch_all[i][key]]
            lst_all.append(lst)

        export_all = pd.DataFrame(lst_all, index=indices, columns=tuplekeys_all)