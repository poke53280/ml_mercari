

from mp4_frames import get_ready_data_dir
from mp4_frames import get_model_dir

from InterpolateModel import predict

from pandas import pd


import pathlib

from sklearn.svm import SVC
import pickle



def line_pairs():

    line_dir = get_ready_data_dir()

    assert line_dir.is_dir()

    line_files = list (sorted(line_dir.iterdir()))

    line_files = [x.name for x in line_files]

    line_files = [x[2:] for x in line_files]

    cluster = [x.split("_")[0] for x in line_files]

    name = [x.split("_")[1] for x in line_files]

    ext = [x[-3:] for x in name]

    name = [x[:-4] for x in name]

    df_c = pd.DataFrame({'name': name, 'cluster':  cluster, 'ext': ext})

    sCount = df_c.groupby('name').size()

    df_c = df_c.assign(numfiles = df_c.name.map(sCount))


    m_single = df_c.numfiles == 1

    print (f"Dropping singles {m_single.sum()}")

    df_c = df_c[~m_single].reset_index(drop = True)

    df_c = df_c.drop(['ext', 'numfiles'], axis = 1)

    m = df_c.duplicated(subset = 'name')

    df_c = df_c[~m].reset_index(drop = True)

    return df_c


def model_pairs():


    model_dir = get_model_dir()

    assert model_dir.is_dir()

    model_files = list (sorted(model_dir.iterdir()))

    model_files = [x for x in model_files if "h5" in str(x)]

    model_files = [x.name for x in model_files]

    model_files = [x[2:] for x in model_files]

    cluster = [x.split("_")[0] for x in model_files]

    name = [x.split("_")[1] for x in model_files]

    realfake = [x.split("_")[2] for x in model_files]

    realfake = [x[:-3] for x in realfake]

    df_m = pd.DataFrame({'name': name, 'cluster': cluster, 'rf' : realfake})

    sCount = df_m.groupby('name').size()

    df_m = df_m.assign(numfiles = df_m.name.map(sCount))


    m_single = df_m.numfiles == 1

    print (f"Dropping singles {m_single.sum()}")

    df_m = df_m[~m_single].reset_index(drop = True)

    df_m = df_m.drop(['rf', 'numfiles'], axis = 1)

    m = df_m.duplicated(subset = 'name')

    df_m = df_m[~m].reset_index(drop = True)

    return df_m

def load_model_pair(model_cluster, model_name):

    real_file = get_model_dir() / f"c_{model_cluster}_{model_name}_real.h5"
    fake_file = get_model_dir() / f"c_{model_cluster}_{model_name}_fake.h5"
    assert real_file.is_file() and fake_file.is_file()

    from keras.models import load_model

    model_real = load_model(real_file)
    model_fake = load_model(fake_file)

    return model_real, model_fake



def predict_lineset(sample_model, sample_lineset):


    model_name = sample_model['name']
    model_cluster = sample_model['cluster']

    model_real, model_fake = load_model_pair(model_cluster, model_name)


    lineset_name = sample_lineset['name']
    lineset_cluster = sample_lineset['cluster']


    df_file = get_ready_data_dir() / f"c_{model_cluster}_{model_name}.pkl"
    assert df_file.is_file()

    npy_file = get_ready_data_dir() / f"c_{model_cluster}_{model_name}.npy"
    assert npy_file.is_file()

    df = pd.read_pickle(df_file)

    data = np.load(npy_file)

    m_fake = (df.fake == True)
    m_real = (df.fake == False)


    # A model and a line set, make prediction with errors

    NUM_CUT = 1000

    err_mr_lr = predict(model_real,  data[m_real][:NUM_CUT])
    err_mf_lr = predict(model_fake,  data[m_real][:NUM_CUT])

    err_mr_lf = predict(model_real,  data[m_fake][:NUM_CUT])
    err_mf_lf = predict(model_fake,  data[m_fake][:NUM_CUT])

    return err_mr_lr, err_mf_lr, err_mr_lf, err_mf_lf


def predict_linesets_against_models(sample_lineset, l_m):


    fake_stat = []
    real_stat = []

    for x in l_m:

        sample_model = {}

        sample_model['name'] = x[1]
        sample_model['cluster'] = x[0]
  

        print (f"Model c{sample_model['cluster']}_{sample_model['name']} predict on lineset c{sample_lineset['cluster']}_{sample_lineset['name']}")

        err_mr_lr, err_mf_lr, err_mr_lf, err_mf_lf = predict_lineset(sample_model, sample_lineset)

        real_stat.append(err_mr_lr)
        real_stat.append(err_mf_lr)

        fake_stat.append(err_mr_lf)
        fake_stat.append(err_mf_lf)

        print (err_mr_lr, err_mf_lr, err_mr_lf, err_mf_lf)


    acReal = np.array(real_stat)
    acReal0 = acReal[::2]
    acReal1 = acReal[1::2]

    acDiffReal = acReal0 - acReal1


    acFake = np.array(fake_stat)
    acFake0 = acFake[::2]
    acFake1 = acFake[1::2]

    acDiffFake = acFake0 - acFake1

    return acDiffReal, acDiffFake



df_l = line_pairs()
df_m = model_pairs()

mod_clusters = np.unique(np.array(df_m.cluster))
line_clusters = np.unique(np.array(df_l.cluster))

m_line_clusters_in_mod_clusters = df_l.cluster.isin(mod_clusters)

df_l = df_l[~m_line_clusters_in_mod_clusters].reset_index(drop = True)

l_l = list(zip(df_l.cluster, df_l.name))

l_m = list(zip(df_m.cluster, df_m.name))

l_m_order = [f"c_{x[0]}_{x[1]}" for x in l_m]


s_name = []
s_cluster = []
s_y = []
s_data = []


for x in l_l:
    sample_lineset = {}
    sample_lineset['name'] = x[1]
    sample_lineset['cluster'] = x[0]

    aT, aF = predict_linesets_against_models(sample_lineset,l_m)

    s_name.append(sample_lineset['name'])
    s_cluster.append(sample_lineset['cluster'])
    s_y.append(True)
    s_data.append(aF)

    s_name.append(sample_lineset['name'])
    s_cluster.append(sample_lineset['cluster'])
    s_y.append(False)
    s_data.append(aT)



df_meta = pd.DataFrame({'cluster': s_cluster, 'name': s_name, 'y': s_y})

df = pd.concat([df_meta, pd.DataFrame(np.stack(s_data))], axis = 1)

df = df.assign(cluster = df.cluster.astype(int))

df.to_pickle(get_model_dir() / "stage2.pkl")


l_cluster_valid = [11, 100, 31, 120, 50,30, 180, 130, 0, 10]

df = df.assign(validation = df.cluster.isin(l_cluster_valid))


y = df.y

validation = df.validation

df = df.drop(['cluster', 'name', 'validation','y'], axis = 1)

X = np.array(df)


X_train = X[~validation]
X_valid = X[validation]

y_train = y[~validation]
y_valid = y[validation]

clf = SVC(gamma='auto')
clf.fit(X_train, y_train)


filename = get_model_dir() / "finalized_model.sav"

pickle.dump(clf, open(filename, 'wb'))

del clf

clf = pickle.load(open(filename, 'rb'))

y_pred = np.array(clf.predict(X_valid))


df_pred = pd.DataFrame({'y': y_valid, 'y_p': y_pred})



