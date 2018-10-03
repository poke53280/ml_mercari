

#
# Walkthrough with decoding stage on:
# https://github.com/tensorflow/tensor2tensor/blob/master/README.md
#
#

# Use SSD disk

t2t-datagen --data_dir=/mnt/disks/tmp_mnt/ende_data --tmp_dir=/mnt/disks/tmp_mnt/ende_tmp --problem=translate_ende_wmt32k


gsutil cp -r ./ende_tmp/ gs://anders_eu/wmt_data


CONTINUE HERE




# Also use TPU/tensor2tensor tutorial:
#
# https://cloud.google.com/tpu/docs/tutorials/transformer
#
# ... for TPU parameters.
#
