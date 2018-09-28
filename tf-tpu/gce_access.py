
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



pip install tensor2tensor

DATA_DIR=$STORAGE_BUCKET/train_data/


Local: Create  path to t2t tools - see location with pip show <package name>.

Local not working (obviously noone has tried anything on windows).

LOCAL APPROACHED DROPPED


-------------------------------------------------------------------------------------------------------------------

CREATE VM DISK

gcloud compute disks create tmpt2t --size 200 --type pd-ssd

gcloud compute instances attach-disk tpu-driver-eur --disk tmpt2t 

sudo lsblk

sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdbXXX

sudo mkdir -p /mnt/disks/tmp_mnt

sudo mount -o discard,defaults /dev/sdb /mnt/disks/tmp_mnt

--------------------------------------------------------------------------------------------------------------------


sudo mkdir /mnt/disks/tmp_mnt/t2t_tmp

TMP_DIR=/mnt/disks/tmp_mnt/t2t_tmp


DATA_DIR=$STORAGE_BUCKET/train_data

sudo chmod 777 -R /mnt/disks/tmp_mnt/

t2t-datagen --problem=translate_ende_wmt32k_packed --data_dir=$DATA_DIR --tmp_dir=$TMP_DIR

DELETE VM DISK

gcloud compute instances detach-disk tpu-driver-eur --disk=tmpt2t
gcloud compute disks delete tmpt2t





OUT_DIR=$STORAGE_BUCKET/training/transformer_ende_1


t2t-trainer --model=transformer --hparams_set=transformer_tpu --problem=translate_ende_wmt32k_packed --train_steps=10 --eval_steps=3 --data_dir=$DATA_DIR --output_dir=$OUT_DIR --use_tpu=True --cloud_tpu_name=$TPU_NAME


==> AttributeError: 'RunConfig' object has no attribute 'data_parallelism'


