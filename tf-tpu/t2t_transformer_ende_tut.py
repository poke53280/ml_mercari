
#
# WMT DATA
#



#
# Walkthrough including decoding stage on:
# https://github.com/tensorflow/tensor2tensor/blob/master/README.md
# (Not TPU specific)

# Use SSD disk

t2t-datagen --data_dir=/mnt/disks/tmp_mnt/ende_data --tmp_dir=/mnt/disks/tmp_mnt/ende_tmp --problem=translate_ende_wmt32k


gsutil cp -r ./ende_tmp/ gs://anders_eu/wmt_data

gsutil cp -r ./ende_data/ gs://anders_eu/wmt_datadata


# Rename folder on gs

gsutil mv gs://anders_eu/wmt_data gs://anders_eu/wmt_tmp
gsutil mv gs://anders_eu/wmt_datadata gs://anders_eu/wmt_data



DATA_DIR=${STORAGE_BUCKET}/wmt_data/wmt_datadata
MODEL_DIR=${STORAGE_BUCKET}/wmt_model/





t2t-trainer --data_dir=$DATA_DIR --problem=translate_ende_wmt32k --hparams_set=transformer_tpu --model=transformer --output_dir=$MODEL_DIR --train_steps=10 --eval_steps=1 --use_tpu=True --cloud_tpu_name=$TPU_NAME

#train steps: 2000
#~20 minutes

### DECODING ###

# nano decode_this.txt
  # enter english text

# gsutil cp decode_this.txt $DATA_DIR

t2t-decoder --data_dir=$DATA_DIR --problem=translate_ende_wmt32k --hparams_set=transformer_tpu --model=transformer --output_dir=$MODEL_DIR --use_tpu=True --cloud_tpu_name=$TPU_NAME --decode_hparams="beam_size=4,alpha=0.6" --decode_from_file=$DATA_DIR/decode_this.txt --decode_to_file=$DATA_DIR/translation.en


# CONTINUE HERE: => ValueError: TPU can only decode from dataset

#
# Data => Dataset
#

# Create dataset 







# Also use TPU/tensor2tensor tutorial:
#
# https://cloud.google.com/tpu/docs/tutorials/transformer
#
# ... for TPU parameters.
#
