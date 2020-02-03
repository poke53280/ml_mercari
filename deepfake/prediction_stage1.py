


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

    # create random sequence as baseline
    #y_random = np.random.uniform(size = test_sequence_real.shape)
    #y_random = y_random.reshape((-1, 16, 3))


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
    
    print(f"Mse error {mse}")
    return mse



p = get_output_dir()
assert p.is_dir()

model = load_model(p / 'my_model.h5')

l_mse = []
l_is_real = []

for i in range (10):

    isReal = np.random.choice([True, False])

    video_base = get_test_video(32, isReal)

    aS = sample_video_outer(video_base)

    mse = predict(model, preprocess_input(aS))
    
    l_is_real.append(isReal)
    l_mse.append(mse)

df = pd.DataFrame({'mse' : l_mse, 'y' : l_is_real})


df


