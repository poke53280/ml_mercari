
import sys
import pandas as pd
import numpy as np

DATA_DIR_PORTABLE = "C:\\plasticc_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE


training = pd.read_csv(DATA_DIR + 'training_set.csv')
meta_training = pd.read_csv(DATA_DIR + 'training_set_metadata.csv')

print("Dataset loaded")

IDs = meta_training.object_id

assert IDs.shape == np.unique(IDs).shape


def split_and_save(filename_desc, iSplit, num_split):
   
    assert iSplit >= 0 and iSplit < num_split

    IDsplit = np.array_split(IDs, num_split)

    anIDslice = IDsplit[iSplit]

    m = training.object_id.isin(anIDslice)

    pd.to_pickle(training[m], DATA_DIR + filename_desc + "_training_" + str(iSplit) + ".pkl")

    m = meta_training.object_id.isin(anIDslice)

    pd.to_pickle(meta_training[m], DATA_DIR + filename_desc + "_meta_" + str(iSplit) + ".pkl")



def main():
    # print command line arguments
    l = []
    for arg in sys.argv[1:]:
        l.append(arg)

    desc = l[0]
    num_splits = int (l[1])

    print(f"File desc: {desc}")
    print(f"Num splits: {num_splits}")

    assert num_splits > 0

    for isplit in range(num_splits):
        split_and_save(desc, isplit, num_splits)


if __name__ == "__main__":
    main()






