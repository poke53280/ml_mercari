

import numpy as np
from pathlib import Path
from mp4_frames import get_output_dir
from mp4_frames import get_ready_data_dir
import matplotlib.pyplot as plt



output_dir = get_output_dir()
ready_dir = get_ready_data_dir()


l_files = list (sorted(output_dir.iterdir()))

l_files = [x for x in l_files if "npy" in x.suffix]

iFile = 0

photos = list()
labels = list()

for x in l_files:

    print (x)
    anData = np.load(x)

    video_size = 32

    W = 256
    H = 1

    anData = anData.reshape(-1, video_size, W, 3)

    anReal = anData[:7]
    anFake = anData[7:14]

    for i in range(7):
        photos.append(anReal[i])
        labels.append(0.0)
        photos.append(anFake[i])
        labels.append(1.0)

    isLast = (x == l_files[-1])
    
    if isLast or len(photos) > 1000:
        photos = np.asarray(photos)
        labels = np.asarray(labels)
        photos = photos/255.0

        filepath_photo = ready_dir / f"photos_{iFile:04}.npy"
        filepath_label = ready_dir / f"labels_{iFile:04}.npy"

        np.save(filepath_photo, photos)
        np.save(filepath_label, labels)

        iFile = iFile + 1
        photos = list()
        labels = list()



def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 256, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model



def later():

    num_all = photos.shape[0]
    num_test = 5
    num_train = num_all - num_test

    idx = np.array(range(num_all))
    idx_test = np.random.choice(idx, size = num_test, replace = False)
    idx_train = np.array(list(set(idx)  - set(idx_test)))

    anTrain = photos[idx_train]
    anTest  = photos[idx_test]

    anYTrain = labels[idx_train]
    anYTest  = labels[idx_test]

    model = define_model()

    history = model.fit(x = anTrain, y = anYTrain, batch_size = 64, epochs = 20, verbose = 1, validation_data=[anTest, anYTest])


