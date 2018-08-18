#!/usr/bin/env sh
# script for performing image retrieval using features stored in leveldb/lmdb    

# target folders
TARGET_DATASET=UnitTest_Django
TARGET_DB=leveldb
TARGET_IMG_TYPE=JPEG

TARGET_DIR=$HOME/Documents/Dataset/$TARGET_DATASET

# path to retrieval Binary
BIN_EXTRACT=../../build/tools/Wei_retrieval.bin

# the blob name to be extracted
BLOB_NAME=fc7

# run feature extraction on certain dataset
$BIN_EXTRACT $TARGET_DIR/$TARGET_DB/feature_$BLOB_NAME $TARGET_DB
