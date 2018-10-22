

#def - merge and save




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

    df = pd.DataFrame()

    for i in range (num_splits):

        filename = DATA_DIR + "df_t_" + desc + "_" + str(i)+ ".pkl"
        df_load = pd.read_pickle(filename)

        df = pd.concat([df, df_load], axis = 0)

    filename = DATA_DIR + "df_t_" + desc + "_" + "Merged" + ".pkl"

    df.to_pickle(filename)

    print(f"Merged chunks to file: '{filename}'")

    


if __name__ == "__main__":
    main()

