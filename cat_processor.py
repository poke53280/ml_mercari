




"""
In Dataframe q, my_cat category, find element closest to having -input- value in brand_name
"""

def get_cutoff(series, count_threshold):
    q = series.value_counts()
    q = q.reset_index()

    cut_index = q.ix[ (q[q.columns[1]] - count_threshold).abs().argsort()[:1]]
    actual_cut_index = cut_index.index[0]
    num_categories = actual_cut_index

    return num_categories

"""--------------------------------------------------------------------------------------"""


"""

In train: brand freq

 500 => 122 items
1000 =>  33 items
1500 =>  14 items
2000 =>   7 items
3000 =>   3 items

"""




"""Category analysis"""


"""Number of items in each final category"""
q = train.category_name.value_counts()

df = pd.DataFrame(q.index, q)

df.reset_index(inplace=True)

df = df[[0, 'category_name']]

df.columns = ['cat', 'freq']

cat_and_freq = df

cat_and_freq['cat_0'], cat_and_freq['cat_1'], cat_and_freq['cat_2'] = zip(*cat_and_freq['cat'].apply(lambda x: split_cat(x)))

"""Remove full cat"""
cat_and_freq = cat_and_freq [['cat_0', 'cat_1', 'cat_2', 'freq']]

"""Group small categories together/ merge with bigger categories to a minimum of N entries in each remaining category"""
cat_and_freq.loc[cat_and_freq.freq > 2000]
"""=> 151 rows e.g."""

"""Number of largest cat_2 categories vs items contained"""

N = 1000
num_cats = len(cat_and_freq.loc[cat_and_freq.freq > N])
num_items = cat_and_freq.loc[cat_and_freq.freq > N].freq.sum()

num_items_total = cat_and_freq.freq.sum()

print("N = " + str(N) + ", num_cats = " + str(num_cats) + ", num_items = " + str(num_items) + ", " + str(num_items/ num_items_total))


"""

N >   500,  num_cats = 345,  num_items  = 1396885,  0.9462657023942426
N >  1000,  num_cats = 238,  num_items =  1319123,  0.8935888438485633
N >  4000,  num_cats =  88,  num_items  = 1019533,  0.6906431884937624
N >  6000,  num_cats =  62,  num_items  =  890917,  0.6035172550209726
N > 10000,  num_cats =  36,  num_items  =  689400,  0.4670073593965078

"""

cat_and_freq.loc[ (cat_and_freq.cat_0 == 'Handmade')]

cat_and_freq.loc[ (cat_and_freq.cat_0 == 'Handmade') & (cat_and_freq.cat_1 == 'Patterns')]


cat_and_freq.loc[ (cat_and_freq.cat_0 == 'Handmade') & (cat_and_freq.cat_1 == 'Patterns') & (cat_and_freq.P_Cat == 'Drop')].freq.sum()

""" => 48 in Handmade-Patterns to be processed together """



"""Create marker column"""
  

"""
For item in tiny category, take note of category name in _name_ and/or _description."
Or keep it all, just prepare new flat categories
"""


    