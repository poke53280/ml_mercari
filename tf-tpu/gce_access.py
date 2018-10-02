
# gcloud auth application-default login

import googleapiclient.discovery

compute = googleapiclient.discovery.build('compute', 'v1')

compute.instances().list(project="nav-datalab", zone="europe-west4-a")

q = _

res = q.execute()

res['items']

############################################################### T2T TRANSLATE #####################################################################


Tutorial 


https://cloud.google.com/tpu/docs/tutorials/transformer

git clone https://github.com/tensorflow/tensor2tensor


LOCAL APPROACHED DROPPED - POOR WINDOWS SUPPORT


-------------------------------------------------------------------------------------------------------------------



sudo mkdir /mnt/disks/tmp_mnt/t2t_tmp

TMP_DIR=/mnt/disks/tmp_mnt/t2t_tmp


DATA_DIR=$STORAGE_BUCKET/train_data

sudo chmod 777 -R /mnt/disks/tmp_mnt/

t2t-datagen --problem=translate_ende_wmt32k_packed --data_dir=$DATA_DIR --tmp_dir=$TMP_DIR


OUT_DIR=$STORAGE_BUCKET/training/transformer_ende_1


t2t-trainer --model=transformer --hparams_set=transformer_tpu --problem=translate_ende_wmt32k_packed --train_steps=10 --eval_steps=3
--data_dir=$DATA_DIR --output_dir=$OUT_DIR --use_tpu=True --cloud_tpu_name=$TPU_NAME


==> AttributeError: 'RunConfig' object has no attribute 'data_parallelism'



# START FROM COLD

source .mybashrc
source myscript.sh


DATA_DIR=$STORAGE_BUCKET/train_data/
OUT_DIR=$STORAGE_BUCKET/training/transformer_ende_1


#
# ----------------------------------------------------------------------------------------------------
#
# SOMETHING WORKING:
#
# ON INSTANCE.
# CHECK: RUNNING LOCAL?
#

t2t-trainer 

--generate_data
--data_dir = ~/data/ 
--output_dir = ~/train_mnist 
--problem=image_mnist 
--model=shake_shake 
--hparams_set=shake_shake_quick 
--train_steps=1000
--eval_steps=100

(Stopped)


--hparams_set - competting setups. Check for _tpu configs.



# Try these 'tpu parameters'

DATA_DIR=$STORAGE_BUCKET/imdb/
OUT_DIR=$STORAGE_BUCKET/imdb_out/




t2t-trainer --data_dir=$DATA_DIR --output_dir=$OUT_DIR --problem=sentiment_imdb --model=transformer_encoder --hparams_set=transformer_tiny_tpu --train_steps=10 --eval_steps=3 --use_tpu=True --cloud_tpu_name=$TPU_NAME


# params:
# batch-size 100



# SOMETHING MORE WORKING

# Hard code tpu in code.

t2t-trainer --model=transformer --hparams_set=transformer_tpu --problem=translate_ende_wmt32k_packed --train_steps=1000 --eval_steps=3 --data_dir=$DATA_DIR --output_dir=$OUT_DIR --use_tpu=True --cloud_tpu_name=$TPU_NAME


#
#---------------------------------------------------------------------------
#
# TODO: Find the bug in t2t (missing TPU setting) and post on github.
#
#
#
# Find state of debugging/running from scratch on windows.
#    Or: Debugging on instance.
#
# Check out: Edit local => push instance => Run => Edit loop
#
# Tensor2Tensor - viability
#
# Tensorflow subset. 
#
#




















# FIRST:
t2t-datagen --problem=image_cifar10 --data_dir=~/cifar10_data --tmp_dir=~/temp_cifar10


gsutil cp -r ~/cifar10_data ${STORAGE_BUCKET}/cifar10_data

DATA_DIR=${STORAGE_BUCKET}/cifar10_data


gsutil ls $DATA_DIR

...

delete old output dir


t2t-trainer --model=shake_shake --hparams_set=shakeshake_tpu --problem=image_cifar10 --train_steps=180000 --eval_steps=9 --local_eval_frequency=100 --data_dir=$DATA_DIR --output_dir=$OUT_DIR --use_tpu --cloud_tpu_name=$TPU_NAME



















