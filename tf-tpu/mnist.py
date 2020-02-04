
#
# Using functions from tensorflow.models.official:
#
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Apache licensed, see:  http://www.apache.org/licenses/LICENSE-2.0
#
# ----------------------------------------------------------------------------
#
#
#
#
# TODO PREDICT NUMBERS AND REVIEW RESULT
#
#


# TODO Bring:
# https://github.com/AvoncourtPartners/poems into TPU form.

# Other refs:
# https://medium.com/@DenisKrivitski/generation-of-poems-with-a-recurrent-neural-network-128a5f62b483
# http://karpathy.github.io/2015/05/21/rnn-effectiveness/
#
#

############
#
# Update gcloud
#
#
#gcloud components update
#









#
# Requirement: Install SDK
#
# Requirement: Create bucket in zone eu-west-4
#
# TPU instructions
# ----------------
#
# Local: Launch terminal 
#
#
#
# COMPUTE INSTANCE VM: CREATE
#
# gcloud compute instances create tpu-driver-eur --machine-type=n1-standard-2 --image-project=ml-images --image-family=tf-1-9 --scopes=cloud-platform
# gcloud compute instances create anders-tf1-10 --machine-type=n1-standard-2 --image-project=ml-images --image-family=tf-1-10 --scopes=cloud-platform
#
#
# CREATE VM DISK
#
# gcloud compute disk-types list
# gcloud compute disks create tmpt2t --size 200 --type pd-ssd
# disks create tmpt2t --size 1500 --type pd-standard
#
# gcloud compute instances attach-disk atcompute --disk tmpt2t 
#
# sudo lsblk
#
# sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdbXXX
#
# sudo mkdir -p /mnt/disks/tmp_mnt
#
# sudo mount -o discard,defaults /dev/sdb /mnt/disks/tmp_mnt
#
# sudo chmod 777 -R /mnt/disks/tmp_mnt
#
# DELETE VM DISK
#
# gcloud compute instances detach-disk tpu-driver-eur --disk=tmpt2t
# gcloud compute disks delete tmpt2t
#
# 
#
#
# COMPUTE INSTANCE VM: SETUP
#
#
# https://www.datacamp.com/community/tutorials/google-cloud-data-science
#
#
#sudo apt-get update
#sudo apt-get install bzip2 git libxml2-dev
#
#wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
#
#bash Anaconda3-5.2.0-Linux-x86_64.sh
#rm Anaconda3-5.2.0-Linux-x86_64.sh
#
#
# pip install --upgrade pip
# pip install tensorflow==1.9
#
# pip install --upgrade google-api-python-client
# pip install --upgrade oauth2client
#
#pip install --upgrade cloud-tpu-profiler 
#
# COMPUTE INSTANCE VM: LOGIN
#
# (LOCAL) gcloud compute ssh USERNAME@tpu-driver-eur
#
# TODO - check out integrated ssh on windows: https://www.thomasmaurer.ch/2017/11/install-ssh-on-windows-10-as-optional-feature/
#
# source .bashrc
# source myscript.sh
#
#
# COMPUTE INSTANCE VM: CONFIGURATION
#
# Create a script or run these commands on the command line.
#
#
# (VM):
#
# #!/bin/bash
# gcloud config set compute/region europe-west4
# gcloud config set compute/zone europe-west4-a
# export STORAGE_BUCKET=gs://anders_eu
# export TPU_NAME='preempt-1-9'
#
#
#
#
#
# CHECK ENVIRONMENT VARIABLE
#
# (VM) echo "$TPU_NAME"
# => tpu-anders-eur
#
#
# CREATE TPU
#
#
# gcloud compute tpus create preempt-1-10 --network=default --range=10.240.1.0/29 --version=1.10 --preemptible
# gcloud compute tpus create preempt-1-9 --network=default --range=10.240.1.8/29 --version=1.9 --preemptible
#
# note '8', ,'0' to coexist with neighbouring CIDR addresses.
#
#
#
#
# CHECK TPU STATUS
#
#(VM OR LOCAL) gcloud compute tpus list
#
#
# UPLOAD CODE (*this* very file, mnist.py, and convert_to_records.py)
#
#
# USERNAME often firstname_lastname
#
#(DESKTOP) gcloud compute scp .\mnist.py USERNAME@tpu-driver-eur:.
#(DESKTOP) gcloud compute scp .\convert_to_records.py USERNAME@tpu-driver-eur:.
#
#
# DOWNLOAD DATASET TO INSTANCEVM) python convert_to_records.py --directory=./data
# (VM) gunzip -d on all .gz files.
#
#
# MOVE DATASET FROM INSTANCE TO GS
#
# (VM) gsutil cp -r ./data ${STORAGE_BUCKET}
# (VM) rm -rf ./data/
#
#
# EXECUTE:
#
# (VM) python ./mnist.py --tpu=$TPU_NAME --data_dir=${STORAGE_BUCKET}/data --model_dir=${STORAGE_BUCKET}/output --use_tpu=True --iterations=500 --train_steps=9000
#
#
#
#
# CODE RE-TRANSFER DESKTOP -> VM:
#
# gcloud compute scp .\mnist.py USERNAME@tpu-driver-eur:.
#
# gcloud compute scp .\*.py anders_topper@atcompute:/mnt/disks/tmp_mnt/data

# ... and run again
#
# 
# MOVE OUTPUT FROM GS TO VM
#
# mkdir output 
# gsutil cp -r ${STORAGE_BUCKET}/output .
#


#DELETE OUTPUT IN BUCKET
# gsutil rm ${STORAGE_BUCKET}/output/*

#
# MOVE OUTPUT FROM VM TO LOCAL:
#
# (LOCAL):


# Create folder output
# gcloud compute scp --recurse anders_topper@tpu-driver-eur:./output/* ".\output\"
#
#
# STOP SYSTEM
#
# Log out from VM
#
# (LOCAL OR VM) gcloud compute tpus list
# (LOCAL OR VM) gcloud compute tpus stop tpu-anders-eur 
# 
#
#
# (LOCAL) gcloud compute instances list
# (LOCAL) gcloud compute instances stop tpu-driver-eur
# (LOCAL) gcloud compute instances start tpu-driver-eur
#
#
# RUN LOCAL
#
# Get data - note direct access local to GS possible:
#
# (LOCAL)gsutil cp -r gs://anders_eu/data .
#
# (LOCAL) python ./mnist.py --data_dir=./data --model_dir=./output --use_tpu=False --iterations=500 --train_steps=2006
#
#
#
# COLD START WITH EXISTING RESOURCES
#
# (RE)LAUNCH CLOUD VM AND TPU
# gcloud compute instances list
# gcloud compute tpus list
# gsutil ls
# gsutil ls gs://anders_eu/data
#
# gcloud compute instances start tpu-driver-eur
# gcloud compute tpus start tpu-anders-eur
#
# gcloud compute scp .\mnist.py USERNAME@tpu-driver-eur:.
#
# gcloud ssh...
# Run exports and config zone/region.
#
# gsutil rm gs://anders_eu/output/*
#
# TENSORBOARD
#
# point to output folder, same local and tpu.
#

#
#
# Note: You can either capture a profile or monitor your job; you cannot do both at the same time.
#
# CLOUD TPU TOOLS
# https://cloud.google.com/tpu/docs/cloud-tpu-tools
#
#
# MONITORING YOUR JOB


# gcloud compute ssh anders_topper@tpu-driver-eur --ssh-flag="-L 6006:localhost:6006"









# -----------------------------------------------------------------------------------------
# TODO 
#
# + 1. deploy image model.
# + 2. deploy txt analysis model.
# + 3. deploy GAN model.
#
# + Preemptible setup
#
#



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np

import tensorflow as tf  # pylint: disable=g-bad-import-order


#################################################################################
#
#   readU32
#
# Read 4 bytes from bytestream as an unsigned 32-bit integer
#

def readU32(bytestream):
  
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

#################################################################################
#
#   is_image_file_header_valid
#
# Validate that filename corresponds to images for the MNIST dataset.
#

def is_image_file_header_valid(filename):
  
  with tf.gfile.Open(filename, 'rb') as f:

    magic = readU32(f)

    num_images_unused = readU32(f)  

    rows = readU32(f)
    cols = readU32(f)

    if magic != 2051:
      print('Invalid magic number')
      return False

    if rows != 28 or cols != 28:
      print('Invalid MNIST file : Expected 28x28 images, found x')
      return False

  return True


#################################################################################
#
#   is_labels_file_header_valid
#
# Validate that filename corresponds to labels for the MNIST dataset.
#

def is_labels_file_header_valid(filename):
  
  with tf.gfile.Open(filename, 'rb') as f:

    magic = readU32(f)

    num_items_unused = readU32(f) 

    if magic != 2049:
      print('Invalid magic number {magic} in MNIST file {f.name}')
      return False

  return True    


#################################################################################
#
#   decode_image
#
#   Normalize from [0, 255] to [0.0, 1.0]
#

def decode_image(image):
   
    image = tf.decode_raw(image, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [784])
    return image / 255.0

#################################################################################
#
#   decode_label
#
#

def decode_label(label):
    label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
    label = tf.reshape(label, [])  # label is a scalar
    return tf.to_int32(label)


#################################################################################
#
#   get_input_pipeline
#
#

def get_input_pipeline(directory, images_file, labels_file):
  """Parse MNIST dataset."""

  images_file = os.path.join(directory, images_file)
  assert tf.gfile.Exists(images_file), "image data not downloaded"

  labels_file = os.path.join(directory, labels_file)
  assert tf.gfile.Exists(labels_file), "image data not downloaded"

  assert is_image_file_header_valid(images_file), "Invalid image file(s)"
  assert is_labels_file_header_valid(labels_file), "Invalid labels file(s)"

  images = tf.data.FixedLengthRecordDataset(images_file, 28 * 28, header_bytes=16).map(decode_image)
  labels = tf.data.FixedLengthRecordDataset(labels_file, 1, header_bytes=8).map(decode_label)

  return tf.data.Dataset.zip((images, labels))


# Cloud TPU Cluster Resolver flags
tf.flags.DEFINE_string("tpu", default=None,
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")
tf.flags.DEFINE_string(
    "tpu_zone", default=None,
    help="[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
tf.flags.DEFINE_string(
    "gcp_project", default=None,
    help="[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

# Model specific parameters

# Set defaults to local settings. Issue params on TPU.

tf.flags.DEFINE_string("data_dir", "", "Path to directory containing the MNIST dataset")
tf.flags.DEFINE_string("model_dir", "", "Estimator model_dir")
tf.flags.DEFINE_integer("batch_size", 1024, "Mini-batch size for the training. Note that this is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer("train_steps", 1000, "Total number of training steps.")
tf.flags.DEFINE_integer("eval_steps", 0, "Total number of evaluation steps. If `0`, evaluation after training is skipped.")
tf.flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")

tf.flags.DEFINE_bool("use_tpu", False, "Use TPUs rather than plain CPUs")
tf.flags.DEFINE_integer("iterations", 50, "Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")

FLAGS = tf.flags.FLAGS


#################################################################################
#
#   metric_fn
#
#

def metric_fn(labels, logits):
  accuracy = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))
  return {"accuracy": accuracy}


#################################################################################
#
#   create_model_keras
#
#

def create_model_keras(data_format):
  """
  Network structure is equivalent to:
  https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/examples/tutorials/mnist/mnist_deep.py
  and
  https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py

  But uses the tf.keras API.

  Args:
    data_format: Either 'channels_first' or 'channels_last'. 'channels_first' is
      typically faster on GPUs while 'channels_last' is typically faster on
      CPUs. See
      https://www.tensorflow.org/performance/performance_guide#data_formats

  Returns:
    A tf.keras.Model.
  """
  if data_format == 'channels_first':
    input_shape = [1, 28, 28]
  else:
    assert data_format == 'channels_last'
    input_shape = [28, 28, 1]

  l = tf.keras.layers
  max_pool = l.MaxPooling2D(
      (2, 2), (2, 2), padding='same', data_format=data_format)
  # The model consists of a sequential chain of layers, so tf.keras.Sequential
  # (a subclass of tf.keras.Model) makes for a compact description.
  return tf.keras.Sequential(
      [
          l.Reshape(
              target_shape=input_shape,
              input_shape=(28 * 28,)),
          l.Conv2D(
              32,
              5,
              padding='same',
              data_format=data_format,
              activation=tf.nn.relu),
          max_pool,
          l.Conv2D(
              64,
              5,
              padding='same',
              data_format=data_format,
              activation=tf.nn.relu),
          max_pool,
          l.Flatten(),
          l.Dense(1024, activation=tf.nn.relu),
          l.Dropout(0.4),
          l.Dense(10)
      ])

#################################################################################
#
#   model_fn
#
#   model_fn constructs the ML model used to predict handwritten digits.
#
#

def model_fn(features, labels, mode, params):
  

  del params
  if mode == tf.estimator.ModeKeys.PREDICT:
    raise RuntimeError("mode {} is not supported yet".format(mode))

  image = features

  if isinstance(image, dict):
    image = features["image"]

  model = create_model_keras("channels_last")

  logits = model(image, training=(mode == tf.estimator.ModeKeys.TRAIN))
  
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  if mode == tf.estimator.ModeKeys.TRAIN:
    
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, tf.train.get_global_step(), decay_steps=100000, decay_rate=0.96)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    if FLAGS.use_tpu:
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, loss=loss, train_op=optimizer.minimize(loss, tf.train.get_global_step()))

  if mode == tf.estimator.ModeKeys.EVAL:
    
    return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, loss=loss, eval_metrics=(metric_fn, [labels, logits]))


#################################################################################
#
#   train_input_fn
#
# train_input_fn defines the input pipeline used for training.
#
# Retrieves the batch size for the current shard. The # of shards is
# computed according to the input pipeline deployment. See
# `tf.contrib.tpu.RunConfig` for details.
#


def train_input_fn(params):
  
  batch_size = params["batch_size"]
  data_dir = params["data_dir"]

  ds = get_input_pipeline(data_dir, 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte')

  ds = ds.cache()
  ds = ds.repeat()
  ds = ds.shuffle(buffer_size=50000)

  # 1.10: ds = ds.batch(batch_size, drop_remainder=True)
  ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))


  images, labels = ds.make_one_shot_iterator().get_next()

  return images, labels

#################################################################################
#
#   eval_input_fn
#
#

def eval_input_fn(params):
  
  batch_size = params["batch_size"]
  data_dir = params["data_dir"]
  
  ds = get_input_pipeline(data_dir, 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')

  # 1.10: ds = ds.batch(batch_size, drop_remainder=True)
  ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

  
  images, labels = ds.make_one_shot_iterator().get_next()

  return images, labels


#################################################################################
#
#   main()
#
#

def main(argv):
  del argv  # Unused.
  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.use_tpu:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    my_tpu_config = tf.contrib.tpu.TPUConfig(FLAGS.iterations, FLAGS.num_shards)
  else:
    tpu_cluster_resolver = None
    my_tpu_config = None

  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      tpu_config=my_tpu_config
  )

  estimator = tf.contrib.tpu.TPUEstimator(
      model_fn=model_fn,
      use_tpu=FLAGS.use_tpu,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size,
      params={"data_dir": FLAGS.data_dir},
      config=run_config
  )


  estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)

  # TPUEstimator.evaluate *requires* a steps argument.
  
  # Note that the number of examples used during evaluation is
  # --eval_steps * --batch_size.
  # So if you change --batch_size then change --eval_steps too.

  if FLAGS.eval_steps > 0:
      d = estimator.evaluate(input_fn=eval_input_fn, steps=FLAGS.eval_steps)
      print('Number of examples used during evaluation:{0}*{1}={2}'.format(FLAGS.eval_steps, FLAGS.batch_size, FLAGS.eval_steps * FLAGS.batch_size))
      print('accuracy: {0}, loss: {1}, global_step: {2}'.format(d['accuracy'], d['loss'], d['global_step']))



if __name__ == "__main__":
  tf.app.run()
