

 
def full_sum(df):
    l_b = list()

    for x in df.b:
        l_b.append(x)

    l_c = list()

    for x in df.c:
        l_c.append(x)

    t = list (tuple(zip (l_b, l_c)))

    return t
"""c"""


df = pd.DataFrame( {'a':['A','A','B','B','B','C'], 'b':[1,2,5,5,4,6], 'c': [9,5,6,4,2,5]})

q = df.groupby('a').apply(full_sum)


q

