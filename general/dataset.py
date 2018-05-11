
import numpy as np

#######################################################
#
#   dataset_create_y_large_a_regression
#

def dataset_create_y_large_ac_small_b_regression(l):

    y = []

    for line in l:
        count_a = line.count('a')
        count_b = line.count('b')
        count_c = line.count('c')

        secret_score = 1.4 * count_a + 1.9 * count_c - 0.9 * count_b

        y.append(secret_score)

    y = np.array(y)
    y = y.astype(np.float32)
    return y

"""c"""


#######################################################
#
#   dataset_create_y_large_a
#

def dataset_create_y_large_a(l, nThreshold):

    y = []

    for line in l:
        count = line.count('a')

        if count >= nThreshold:
            y.append(1.0)
        else:
            y.append(0.0)

    y = np.array(y)
    y = y.astype(np.float32)
    return y

"""c"""


#######################################################
#
#   dataset_create_y_a_first
#

def dataset_create_y_a_first(l):

    y = []

    for line in l:
        if line[0] == 'a':
            y.append(1.0)
        else:
            y.append(0.0)

    y = np.array(y)
    y = y.astype(np.float32)
    return y

"""c"""

#######################################################
#
#   dataset_create_y_contains_d
#

def dataset_create_y_contains_d(l):

    y = []

    for line in l:
        count = line.count('d')
        
        if count == 0:
            y.append(0.0)
        else:
            y.append(1.0)

    y = np.array(y)
    y = y.astype(np.float32)
    return y

"""c"""

############################################################
#
#  dataset_get_basic_sequence
#

def dataset_get_basic_sequence():

    l = []
    l.append("abaa---aa")
    l.append("ab--bbbaa")
    l.append("aa-bbbbaa")
    l.append("ba-a-aaaa")
    l.append("bbbbbaaab")
    l.append("ab--bbbbb")
    l.append("bb-bbbbaa")
    l.append("aa-bbbbaa")
    l.append("aba-bbbba")
    l.append("bbaabbbaa")
    l.append("aabbbabba")
    l.append("baa-bbbaa")
    l.append("aba-bbbaa")
    l.append("ba-b--aba")
    l.append("a-babbbaa")
    l.append("abbbbaaab")
    l.append("aab-aaaba")
    l.append("bbbbaaabb")
    l.append("ab--bbabb")
    l.append("bb-ababaa")
    l.append("aa-babbaa")
    l.append("abaab-bba")
    l.append("bbaabbbaa")
    l.append("aabqbabba")
    l.append("baa-abbaa")
    l.append("baba-bbaa")
    l.append("ab-a--aab")
    l.append("ab--bbbaa")
    l.append("aacbebbaa")
    l.append("ba-a-aaaa")
    l.append("bbabbaaab")
    l.append("ab-bbgbbb")
    l.append("bb-bbbbaa")
    l.append("aab-bbbaa")
    l.append("aba-bbbba")
    l.append("bbaabbbaa")
    l.append("aabbbabba")
    l.append("baa-bbbaa")
    l.append("-aaabbbaa")
    l.append("ba-d--aba")
    l.append("a-babbbaa")
    l.append("abbbcbaaa")
    l.append("aab-aaaba")
    l.append("bbbbababa")
    l.append("ab--bbabb")
    l.append("bb-ababaa")
    l.append("aa-babbaa")
    l.append("ababa-bba")
    l.append("bbaaabbaa")
    l.append("aabcbbaba")
    l.append("baaa-baaa")
    l.append("babbba-aa")
    l.append("abaa---aa")
    l.append("ab--bbbaa")
    l.append("aa-bbbbaa")
    l.append("ba-a-aaaa")
    l.append("bbbbbaaab")
    l.append("ab--bbbbb")
    l.append("bb-bbbbaa")
    l.append("aa-bbbbaa")
    l.append("aba-bbbba")
    l.append("bbaabbbaa")
    l.append("aabbbabba")
    l.append("baa-bbbaa")
    l.append("aba-bbbaa")
    l.append("ba-b--aba")
    l.append("a-babbbaa")
    l.append("abbbbaaab")
    l.append("aab-aaaba")
    l.append("bbbbaaabb")
    l.append("ab--bbabb")
    l.append("bb-ababaa")
    l.append("aa-babbaa")
    l.append("abaab-bba")
    l.append("bbaabbbaa")
    l.append("baabqbabb")
    l.append("abaa-abba")
    l.append("fdfbaba-b")
    l.append("fab-a--aa")
    l.append("adfab--bb")
    l.append("aacadfbeb")
    l.append("ba-dfa-aa")
    l.append("bbabbaaab")
    l.append("ab-bebgbb")
    l.append("bb-bbbadf")
    l.append("aab-bbasd")
    l.append("aba-bbbba")
    l.append("basdfbaab")
    l.append("aasdfabbb")
    l.append("badsfaa-b")
    l.append("dfa-aaabb")
    l.append("badsfa-d-")
    l.append("a-adsfbab")
    l.append("dafabbbcb")
    l.append("fdafaab-a")
    l.append("bbadfbbab")
    l.append("ab--adfbb")
    l.append("bb-aasdfb")
    l.append("aa-asdfba")
    l.append("ababa-baa")
    l.append("bbaaaafab")
    l.append("aabcbbaaa")
    l.append("baaa-baaa")
    l.append("babbaa-aa")

    l.append("abaa---aa")
    l.append("ab--bxbaa")
    l.append("aa-bbbbaa")
    l.append("ba-a-aaaa")
    l.append("bbbbbaaab")
    l.append("ab--bbbbb")
    l.append("bb-bbbbaa")
    l.append("aa-bbbbaa")
    l.append("aba-bbbba")
    l.append("bbaabbbaa")
    l.append("aabbbabba")
    l.append("baa-bbbaa")
    l.append("aba-bbbaa")
    l.append("ba-b--aba")
    l.append("a-babbbaa")
    l.append("abbbbaaab")
    l.append("aab-aaaba")
    l.append("bbbbaaabb")
    l.append("ab--bbabb")
    l.append("bb-ababaa")
    l.append("aa-babbaa")
    l.append("abaab-bba")
    l.append("bbaabbbaa")
    l.append("aabqbxbba")
    l.append("baa-abbaa")
    l.append("baba-bbaa")
    l.append("ab-a--aab")
    l.append("ab--bbbaa")
    l.append("aacbecbaa")
    l.append("ba-a-aaaa")
    l.append("bbabbaaab")
    l.append("ab-bbdbbb")
    l.append("bb-bbbbaa")
    l.append("aab-bbbaa")
    l.append("aba-bbbba")
    l.append("bbaabbbaa")
    l.append("aabbbabba")
    l.append("baa-bbbaa")
    l.append("-aaabbbaa")
    l.append("ba-d--aba")
    l.append("a-babbbaa")
    l.append("abbbcbaaa")
    l.append("aab-aaaba")
    l.append("bbbbababa")
    l.append("ab--bbabb")
    l.append("bb-ababaa")
    l.append("aa-babbaa")
    l.append("ababa-bba")
    l.append("bbaaabbaa")
    l.append("aabcbbaba")
    l.append("baaa-baaa")
    l.append("babbba-aa")
    l.append("abaa---aa")
    l.append("ab--bbbaa")
    l.append("aa-bbbbaa")
    l.append("ba-a-aaaa")
    l.append("bbbbbaaab")
    l.append("ab--bbbbb")
    l.append("bb-bbbbaa")
    l.append("aa-bbbbaa")
    l.append("aba-bbbba")
    l.append("bbaabbbaa")
    l.append("aabbbabba")
    l.append("baa-bbbaa")
    l.append("aba-bbbaa")
    l.append("ba-b--aba")
    l.append("a-babbbaa")
    l.append("abbbbaaab")
    l.append("aab-aadba")
    l.append("bbbbaaabb")
    l.append("ab--bbabb")
    l.append("bb-ababaa")
    l.append("aa-babbaa")
    l.append("abaab-bba")
    l.append("bbaabbaaa")
    l.append("baabqbabb")
    l.append("aaaa-abba")
    l.append("fdfaaba-b")
    l.append("fab-a--aa")
    l.append("adfab--bb")
    l.append("aacadfxeb")
    l.append("ba-dfa-aa")
    l.append("bbabbaaab")
    l.append("ab-bebgbb")
    l.append("bb-babadf")
    l.append("aab-bbasd")
    l.append("aba-bbbba")
    l.append("basdfbaab")
    l.append("aasdfabbb")
    l.append("badsfaa-b")
    l.append("dfa-aaabb")
    l.append("badsfa-d-")
    l.append("a-adsfbab")
    l.append("dafabbbcb")
    l.append("fdafaab-a")
    l.append("bbadfbbab")
    l.append("ab--adfbb")
    l.append("bb-aasdfb")
    l.append("aa-asdfba")
    l.append("ababa-baa")
    l.append("bbaaaafab")
    l.append("aabcbbaaa")
    l.append("baxa-baaa")
    l.append("babbax-aa")

    return l
"""c"""



