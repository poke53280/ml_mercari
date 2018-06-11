
#
#
# Check out: https://pythonhosted.org/goslate/
#
# and googletrans
#
#
#
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52557: 
#
# We leveraged Pavel Ostyakov’s idea of machine translations to augment both train and test data sets'
# using French, German, and Spanish translations translated back to English. Given the possibility of information leaks,
# we made sure to have translations stay on the same side of a train-val split as the original comment.
#  For the predictions, we simply averaged the predicted probabilities of the 4 comments (EN, DE, FR, ES).
#
#
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/48038

from textblob import TextBlob


# import from avito;

q = train[:2000]


########################################################################################
#
#    multi_trans
#
#

def multi_trans(txt):

    text = TextBlob(txt)

    text_en = text.translate()
    text_de = text.translate(to = 'de').translate()
    text_fr = text.translate(to = 'fr').translate()
    text_es = text.translate(to = 'es').translate()
    text_nl = text.translate(to = 'nl').translate()
    text_no = text.translate(to = 'no').translate()

    print(f"EN: {text_en}")
    print("---------------------------------------------------------")

    print(f"DE: {text_de}")
    print("---------------------------------------------------------")
    
    print(f"FR: {text_fr}")
    print("---------------------------------------------------------")

    print(f"ES: {text_es}")
    print("---------------------------------------------------------")
    
    print(f"NL: {text_nl}")
    print("---------------------------------------------------------")

    print(f"NO: {text_no}")
    print("---------------------------------------------------------")

"""c"""

# From: 
# https://github.com/PavelOstyakov/toxic/blob/master/tools/extend_dataset.py
# by : https://github.com/PavelOstyakov
#


from joblib import Parallel, delayed
from textblob import TextBlob
from textblob.translate import NotTranslated

import pandas as pd

NAN_WORD = "_NAN_"


def translate(comment, language):
    if hasattr(comment, "decode"):
        comment = comment.decode("utf-8")

    text = TextBlob(comment)
    try:
        text = text.translate(to=language)
        text = text.translate(to="en")
    except NotTranslated:
        pass

    return str(text)


comments_list = q["description"].fillna(NAN_WORD).values

language_list = ['en', 'es', 'de', 'fr', 'nl', 'no']

parallel = Parallel(8, backend="threading", verbose=5)

for language in language_list:
    print('Translate comments using "{0}" language'.format(language))
    translated_data = parallel(delayed(translate)(comment, language) for comment in comments_list)

    s = pd.Series(translated_data)

    s.to_csv(DATA_DIR + "desc_" + language, index=False)

"""c"""        




