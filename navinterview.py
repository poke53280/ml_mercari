
import pandas as pd
import numpy as np

DATA_DIR_BASEMENT = "C:\\Users\\T149900\\Desktop\\nav-ai\\"
DATA_DIR = DATA_DIR_BASEMENT

l = list (range(2018 - 2002))

acYears = np.array(l)
acYears = 2002 + acYears

d = {}

for yr in acYears:
    q = DATA_DIR + str(yr) + ".csv"
    print (q)
    d[yr] = pd.read_csv( q, sep = ';')

"""c"""

d[2011].isco_versjon.value_counts()

d_tekst = {}

d_tekst[2015] = pd.read_csv( DATA_DIR + "tekst-2015.csv", sep = ';')
d_tekst[2016] = pd.read_csv( DATA_DIR + "tekst-2016.csv", sep = ';')
d_tekst[2017] = pd.read_csv( DATA_DIR + "tekst-2017.csv", sep = ';')

#  Memory, dtypes
#  To category/factors


df = pd.read_csv( DATA_DIR + "2015.csv", sep = ';')

m = df.stillingsnummer == 219201504000248

p = df[m]

df_t = pd.read_csv( DATA_DIR + "tekst-2015.csv", sep = ';')

m = df_t.stillingsnummer == 219201504000248

q = df_t[m]

len (df)
len (df_t)





array([ '<p>Enten du er på jakt etter en heltidsjobb eller en ekstrajobb ved siden av skolen, ønsker vi deg velkommen.Vi trenger\xa01 kioskmedarbeidere ,\xa0\xa0som skal betjene kasse, rydding, vareutplassering og\xa0ta imot\xa0bestillinger.</p>\n<p>Vi kan tilby et engasjement med mulighet for fast ansettelse i bedriften.Det er en forutsetning at du innehar gode norskkunnskaper - skriftlig og muntlig.</p>\n<p>\xa0</p>\n<p>NB! Har ingen erfaring så kan du søke praksis, etter\xa0at praksisperioden er over så for du tilbude om engasjement.</p>\n<p>\xa0</p>\n<p><strong>Arbeidsoppgaver:</strong></p>\n<ul>\n<li>Forberede og lage mat</li>\n<li>Lage mat på bestilling\xa0</li>\n<li>Sørge for å opprettholde lik standard på alt som serveres til enhver tid</li>\n<li>Kundebehandling</li>\n</ul>\n<p><br /><strong>Kvalifikasjoner:</strong></p>\n<ul>\n<li>Trives i et høyt tempo og takler et stressende miljø</li>\n<li>Relevant erfaring fra\xa0 bransjen</li>\n<li>Ha gode samarbeidsevner</li>\n<li>Like å jobbe i team</li>\n<li>Være positiv, lojal og pliktoppfyllende</li>\n<li>Snakke grunnleggende norsk og engelsk\xa0</li>\n</ul>\n<p>\xa0</p>\n<p>Sted: Bærum.<br /><br /> Ansettelsesforhold: Engasjement / Praksis</p>\n<p>Tiltredelse: Snarest,</p>\n<p>Kontakt: # ( kl.13-16)</p>'], dtype=object)











