

from keras.preprocessing.text import Tokenizer
import regex as re

txt = ["hello, there, I am more than 20 years old. 3421: hello 99."]

t = Tokenizer(char_level=False)

t.fit_on_texts(txt)

idx_text = t.texts_to_sequences(txt)

d = t.word_index

l = list (d.keys())

l.sort(key = lambda x: -len(x))

txt[0] = txt[0].lower()

for i in l:
    txt[0] = txt[0].replace(i, "X")

matchList = re.findall(r"[^X]+", txt[0])


d
idx_text

reverse_map = dict(zip(d.values(), d.keys()))

l_words = []

for i in idx_text[0]:
    l_words.append(reverse_map[i])

c = l_words + matchList

c[::2] = l_words
c[1::2] = matchList


out = "".join(c)









