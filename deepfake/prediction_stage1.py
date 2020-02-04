


from numpy import array
from keras.models import load_model
from mp4_frames import get_output_dir
import datetime

from dae_lstm import load_part
from dae_lstm import preprocess_input
from dae_lstm import reconstruction_error

from sklearn.metrics import mean_squared_error

from face_detector import get_test_video
from face_detector import sample_video_outer
import matplotlib.pyplot as plt


####################################################################################
#
#   analyse_test_part_24
#

def analyse_test_part_24():


    p = get_output_dir()

    assert p.is_dir()

    model = load_model(p / 'my_model.h5')

    anTest, l_test_orig = load_part(24)

    assert anTest.shape[0] == len (l_test_orig)

    anTest = preprocess_input(anTest)

    test_sequence_real  = anTest[:, :16, :]
    test_sequence_fake  = anTest[:, 16:, :]


    num_timesteps = test_sequence_real.shape[1]

    azOrig = np.array(l_test_orig)

    azOrigUnique = np.unique(azOrig)
    aiOrig = np.searchsorted(azOrigUnique, azOrig)


    for iOrig in range (np.max(aiOrig)):

        m_0 = aiOrig == iOrig

        iOrig_real = test_sequence_real[m_0]
        iOrig_fake = test_sequence_fake[m_0]

        num_test = 1000

        mse_real = reconstruction_error(model, iOrig_real[:num_test])
        mse_fake = reconstruction_error(model, iOrig_fake[:num_test])

        print(f"{iOrig}: real {mse_real:.3f} fake {mse_fake:.3f}")






####################################################################################
#
#   predict
#

def predict(model, data):

    assert data.shape[1:] == (16, 3)
    assert data.min() >= -1 and data.max() <= 1

    mse = reconstruction_error(model, data)
    
    # print(f"Mse error {mse}")
    return mse



p = get_output_dir()
assert p.is_dir()

model_rr = load_model(p / 'my_model_rr.h5')
model_ff = load_model(p / 'my_model_ff.h5')

l_mse_rr = []
l_mse_ff = []
l_is_real = []
l_mean_rf = []

num_runs = 3

for i in range (num_runs):

    print(f"{i}/ {num_runs}")

    isReal = np.random.choice([True, False])

    print (f"isReal: {isReal}")

    video_base = get_test_video(32, isReal)

    aS = sample_video_outer(video_base)

    data_in = preprocess_input(aS)

    mse_rr = predict(model_rr, data_in)
    print(f"mse_rr: {mse_rr}")
    l_mse_rr.append(mse_rr)

    data_p_rr = model_rr.predict(data_in)
    distro_plot(data_in, data_p_rr)

    mse_ff = predict(model_ff, data_in)
    print(f"mse_ff: {mse_ff}")
    l_mse_ff.append(mse_ff)

    data_p_ff = model_ff.predict(data_in)
    distro_plot(data_in, data_p_ff)

    diff_r_f = np.abs(data_p_rr - data_p_ff)

    diff_r_f = np.sum(diff_r_f, axis = 2)
    diff_r_f = np.sum(diff_r_f, axis = 1)

    l_mean_rf.append(np.mean(diff_r_f))
    
    l_is_real.append(isReal)
    

df = pd.DataFrame({'mse_rr' : l_mse_rr, 'mse_ff' : l_mse_ff, 'mean_rf' : l_mean_rf, 'y' : l_is_real})


df.sort_values(by = 'mean_rf')




#############

data_in = preprocess_input(aS)
data_p = model_rr.predict(data_in)

distro_plot(data_in, data_p)


def distro_plot(data_in, data_p):

    err = (data_p - data_in)
    err = np.abs(err)
    err = np.sum(err, axis = 2)
    err = np.sum(err, axis = 1)

    plt.hist(err, bins = 200)
    plt.show()