

############################################################################################################
#
#
# TRAIN IMDB SENTIMENT ON TENSOR2TENSOR. SETUP FROM SCRATCH.
#
# gcloud compute instances list
# gcloud compute instances start XX
#
# gcloud compute tpus list
# gcloud compute tpus start preempt-1-9
#
# VM: 
# gcloud compute ssh anders_topper@tpu-driver-eur
# VM:
# source .bashrc
# source .myscript


# VM:
# pip Install tensor2tensor '-e' for local edit install.
#
# VM Edit code:
# # Edit tensor2tensor code
# t2t_model.py: estimator_model_fn():
#
#    Define and set use_tpu like this:
#       use_tpu = config and config.use_tpu
#

# Create VM SSD disk.

# VM envirnoment:

# DATA_DIR=$STORAGE_BUCKET/imdb/
# TMP_DIR=/mnt/disks/mnt-dir/t2t_tmp



#
# Run t2t-datagen locally or on VM.
# t2t-datagen --problem=sentiment_imdb --data_dir=$DATA_DIR --tmp_dir=$TMP_DIR

#
# Shut down ssd disk if needed.
#

# Verify:
# gsutil ls -l $DATA_DIR
#
# 
#

# OUT_DIR=$STORAGE_BUCKET/imdb_out/

# t2t-trainer --data_dir=$DATA_DIR --output_dir=$OUT_DIR --problem=sentiment_imdb --model=transformer_encoder --hparams_set=transformer_tiny_tpu --train_steps=10 --eval_steps=1 --use_tpu=True --cloud_tpu_name=$TPU_NAME


TODO: t2t-decoder 


##############################################################################################

