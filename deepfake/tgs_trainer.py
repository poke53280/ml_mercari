


# Courtesy https://www.kaggle.com/meaninglesslives/using-resnet50-pretrained-model-in-keras

from mp4_frames import get_ready_data_dir
from mp4_frames import get_model_dir
from mp4_frames import get_meta_dir

import numpy as np
import pandas as pd
from skimage.transform import resize

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

import tensorflow as tf
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.models import save_model


import gc
gc.enable()


isDraw = False

from keras import backend as K

from tgs_model_funcs import get_unet_resnet
from tgs_model_funcs import bce_dice_loss
from tgs_model_funcs import my_iou_metric


####################################################################################
#
#   display
#

def display(anData, anMask, sCoverage):

    sCoverage = sMask.map(np.sum) / pow(img_size_ori, 2)

    def cov_to_class(val):    
        for i in range(0, 11):
            if val * 10 <= i :
                return i

    sCoverageClass = sCoverage.map(cov_to_class)

    fig, axs = plt.subplots(1, 2, figsize=(15,5))
    sns.distplot(sCoverage, kde=False, ax=axs[0])
    sns.distplot(sCoverageClass, bins=10, kde=False, ax=axs[1])
    plt.suptitle("Salt coverage")
    axs[0].set_xlabel("Coverage")
    axs[1].set_xlabel("Coverage class")

    plt.show()

    max_images = 30
    grid_width = 10
    grid_height = int(max_images / grid_width)
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))

    for i in range (max_images):
        img = anData[i].squeeze()
        mask = anMask[i].squeeze()
       
        ax = axs[int(i / grid_width), i % grid_width]
        ax.imshow(img, cmap="Greys")
        ax.imshow(mask, alpha=0.3, cmap="Greens")
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    plt.suptitle("Green: salt. Top-left: coverage class, top-right: salt coverage, bottom-left: depth")

    plt.show()



def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    #res = np.zeros((img_size_target, img_size_target), dtype=img.dtype)
    #res[:img_size_ori, :img_size_ori] = img
    #return res



img_size_ori = 128
img_size_target = 128




input_dir = get_ready_data_dir()

l_files_all = list (input_dir.iterdir())

l_files = [x for x in l_files_all if not "mask" in x.stem]

l_files_mask = [x.stem + "_mask.png" for x in l_files]
l_files_mask = [input_dir / x for x in l_files_mask]

assert len(l_files) == len(l_files_mask)


l_original = [x.stem.split("_")[0] for x in l_files]

df = pd.DataFrame({'file': l_files, 'file_mask': l_files_mask, 'original': l_original})

l_files_stem = [x.stem.split("_")[1] for x in l_files]

df = df.assign(file_stem = l_files_stem)



rValidationSplit = 0.1

azOriginal = np.unique(df.original)

# Todo seed
np.random.shuffle(azOriginal)

num_originals = azOriginal.shape[0]

num_valid = int(1 + (rValidationSplit * num_originals))
num_train = num_originals - num_valid

azTest  = azOriginal[:num_valid]
azTrain = azOriginal[num_valid:]

m_train = df.original.isin(azTrain)
m_test  = df.original.isin(azTest)

assert (m_train ^ m_test).all()

df = df.assign(m_train = m_train, m_test = m_test)

df.to_pickle(get_meta_dir() / "df_tgs.pkl")


idx_train = np.where(m_train)[0]

# Todo: seed
np.random.shuffle(idx_train)


num_max_files_per_run = 7000

num_splits = int(1 + idx_train.shape[0] / num_max_files_per_run)

l_idx_train = np.array_split(idx_train, num_splits)


z_model_name = "my_keras"
checkpoint_path = str(get_model_dir() / f"{z_model_name}.model")


K.clear_session()

model = get_unet_resnet(input_shape=(img_size_target,img_size_target,3))

#from keras.models import load_model
#model = load_model(checkpoint_path)
#add custom object function

model.compile(loss=bce_dice_loss, optimizer='adam', metrics=[my_iou_metric])

model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_my_iou_metric', mode = 'max', save_best_only = True, verbose = 1)

reduce_lr = ReduceLROnPlateau(factor = 0.1, patience = 4, min_lr = 0.00001, verbose = 1)


m_test  = df.original.isin(azTest)


print(f"Loading sDataTest...")
sValidData, sDataTest = load_to_series_rgb(df.file[m_test])

print(f"Loading sMaskTest...")
sValidMask, sMaskTest = load_to_series_grayscale(df.file_mask[m_test])

sValidPair = sValidData & sValidMask

sDataTest = sDataTest[sValidPair].reset_index(drop = True)
sMaskTest = sMaskTest[sValidPair].reset_index(drop = True)


num_test = m_test.sum()

for iTrain, idx_train in enumerate(l_idx_train):

    print(f"{iTrain +1} / {len(l_idx_train)}")

    m_train = np.zeros(shape = df.shape[0], dtype = np.bool)
    m_train[idx_train] = True

    assert (~(m_train & m_test)).all()

    print(f"Loading sDataTrain...")
    sValidData, sDataTrain = load_to_series_rgb(df.file[m_train])

    print(f"Loading sMaskTrain...")
    sValidMask, sMaskTrain = load_to_series_grayscale(df.file_mask[m_train])

    sValidPair = sValidData & sValidMask

    sDataTrain = sDataTrain[sValidPair].reset_index(drop = True)
    sMaskTrain = sMaskTrain[sValidPair].reset_index(drop = True)

   
    num_train = m_train.sum()


    print(f"Concatenating...")
    sData = pd.concat([sDataTest, sDataTrain], axis = 0, ignore_index = True)
    sMask = pd.concat([sMaskTest, sMaskTrain], axis = 0, ignore_index = True)

    print(f"Reshaping...")
    anData = np.array(sData.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 3)
    anMask = np.array(sMask.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)

    sCoverage = sMask.map(np.sum) / pow(img_size_ori, 2)

    #display(anData, anMask, sCoverage)

    x_valid = anData[:num_test]
    x_train = anData[num_test:]

    y_valid = anMask[:num_test]
    y_train = anMask[num_test:]


    #print(f"Augmenting...")

    #x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
    #y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

    epochs = 1
    batch_size = 64

    history = model.fit(x_train, y_train, validation_data=[x_valid, y_valid], epochs=epochs, batch_size=batch_size,
                        callbacks=[model_checkpoint, reduce_lr],shuffle=True,verbose=1)

    save_model(model, str(get_meta_dir() / f"model_{iTrain}"))


