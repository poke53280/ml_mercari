
import sys
import pandas as pd
import numpy as np

DATA_DIR_PORTABLE = "C:\\plasticc_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE



def split_and_save(data, datafile_prefix, meta, metafile_prefix, iSplit, num_splits):

    print(f"Processing split {iSplit +1}/ {num_splits}...")

    IDs = meta.object_id

    assert IDs.shape == np.unique(IDs).shape
  
    assert iSplit >= 0 and iSplit < num_splits

    IDsplit = np.array_split(IDs, num_splits)

    anIDslice = IDsplit[iSplit]

    m = data.object_id.isin(anIDslice)

    pd.to_pickle(data[m], datafile_prefix + "_" + str(iSplit) + ".pkl")

    m = meta.object_id.isin(anIDslice)

    pd.to_pickle(meta[m], metafile_prefix + "_" + str(iSplit) + ".pkl")


def main():
    # print command line arguments
    l = []
    for arg in sys.argv[1:]:
        l.append(arg)

    tr_te = l[0]

    desc = l[1]
    num_splits = int (l[2])
    do_splits = int(l[3])

    assert num_splits > 0, f"num_splits > 0"

    assert do_splits <= num_splits, f"do_splits <= num_splits"

    assert tr_te == "te" or tr_te == "tr"
    isTest = (tr_te == "te")

    if isTest:
        print(f"Test split")
    else:
        print("Train split")

    print(f"File desc: {desc}")
    print(f"Num splits: {num_splits}")
    print(f"Do splits: {do_splits}")

    if isTest:
        data = pd.read_csv(DATA_DIR + 'test_set.csv')
        meta = pd.read_csv(DATA_DIR + 'test_set_metadata.csv')

        datafile_prefix = DATA_DIR + desc + "_test"
        metafile_prefix = DATA_DIR + desc + "_test_meta"

    else:
        data = pd.read_csv(DATA_DIR + 'training_set.csv')
        meta = pd.read_csv(DATA_DIR + 'training_set_metadata.csv')

        datafile_prefix = DATA_DIR + desc + "_training"
        metafile_prefix = DATA_DIR + desc + "_training_meta"
    

    for iSplit in range(do_splits):
        split_and_save(data, datafile_prefix, meta, metafile_prefix, iSplit, num_splits)

    """c"""



if __name__ == "__main__":
    main()






