#!/usr/bin/env sh

# target folders
TARGET_DATASET=INRIA_Holidays
TARGET_DB=leveldb
TARGET_IMG_TYPE=JPEG

TARGET_DIR=$HOME/Documents/Dataset/$TARGET_DATASET

# path to the CNN model Binary
MODEL_BIN_PATH=../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
# path to the CNN model Protocol Text
MODEL_PROTO_PATH=$TARGET_DIR/imagenet_val.prototxt
# path to feature extraction Binary
BIN_EXTRACT=../../build/tools/extract_features_large_scale.bin

# the blob name to be extracted
BLOB_NAME=fc7

# run feature extraction on certain dataset
$BIN_EXTRACT $MODEL_BIN_PATH $MODEL_PROTO_PATH $BLOB_NAME $TARGET_DIR/$TARGET_DB/feature_$BLOB_NAME 1491 $TARGET_DB
