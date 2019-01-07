
import pandas as pd
import regex as re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence

DATA_DIR_PORTABLE = "c:\\names_data\\"
DATA_DIR = DATA_DIR_PORTABLE


# LAST NAME LIST

df = pd.read_csv(DATA_DIR + "etternavn.csv", sep = ';')
df.columns = ['Navn', 'Antall', '?']

s = df.Navn.apply(lambda x: re.sub("\d+\s+", "", x).lower())
lastname_list = list (s.values)


# FIRST NAME GIRLS

df = pd.read_csv(DATA_DIR + "jentanavn_api.csv", sep = ';', encoding = 'latin-1')
s = df.NAVN.apply(lambda x: x.lower())

girlname_list = list (s)


# FIRST NAME BOYS

df = pd.read_csv(DATA_DIR + "guttenavn_api.csv", sep = ';', encoding = 'latin-1')
s = df.NAVN.apply(lambda x: x.lower())

boyname_list = list (s)


# ZIP LOCATIONS

df = pd.read_csv(DATA_DIR + "Postnummerregister-ansi.txt", sep = '\t',encoding = 'latin-1')


s0 = df['B'].apply(lambda x: x.lower())
s1 = df['D'].apply(lambda x: x.lower())

s = set()
s = s.union(list (s0.values))
s = s.union(list (s1.values))

zip_list = list (s)


# FYLKER

county_list = ['Nord-Trøndelag', 'Sør-Trøndelag', 'Østfold', 'Akershus', 'Oslo', 'Hedmark', 'Oppland', 'Buskerud', 'Vestfold', 'Telemark', 'Aust-Agder', 'Vest-Agder', 'Rogaland', 'Hordaland', 'Sogn og Fjordane',
               'Møre og Romsdal', 'Nordland', 'Troms', 'Finmark', 'Trøndelag']

county_list = [x.lower() for x in county_list]



t = pd.Series(["Anders fra Skien sliter med ryggen. Sendt Hansen.", "Vera Henriksen bor i Kragerø i Telemark", "Guro til aap"])




tokenizer = Tokenizer()

tokenizer.fit_on_texts(t)

sequences = tokenizer.texts_to_sequences(t)

print (sequences)

d = tokenizer.word_index

d_reverse = {}

for k, v in d.items():

    if k in zip_list:
        print(f"Warning: {v} in zip list. Remapping")
        d_reverse[v] = "Flåklypa"

    elif k in girlname_list:
        print(f"Warning: {v} in girl name list. Remapping")
        d_reverse[v] = "Kari"
    elif k in boyname_list:
        print(f"Warning: {v} in boy name list. Remapping")
        d_reverse[v] = "Ola"
    elif k in lastname_list:
        print(f"Warning: {v} in last name list. Remapping")
        d_reverse[v] = "Nordmann"
    elif k in county_list:
        d_reverse[v] = "Småland"
    else:
        d_reverse[v] = k
"""c"""


# Modify, print back out ---for samples only---

for iRow in range (len(sequences)):

    line = sequences[iRow]

    list = []

    for x in line:
        list.append(d_reverse[x])

    output = " ".join(list)

    print (output)

"""c"""

# Match saksbehandler.
# Short-date
# Long-date

FID -11- => 22022212345

FullDato0 => Bytt til randomisert offset, felles offset for alle i linje.
KortDato1 => Bytt til randomisert offset, felles offset for alle i linje.

Mnd0

År

Tormod arbeidet i Kragerø i perioden 2002-010399 med smerter i lilletåa. ata2920.020212


Tekst:
Ola arbeidet i Flåklypa i perioden 1601-020299 med smerter i XXX. sak0123. 020312                     

Cat:  A
V1 :  1601
V2 :  020299







Tekst:
D1 etterlyst 010199. 

Cat: E
V1: 010199














































