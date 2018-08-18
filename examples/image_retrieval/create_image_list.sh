#!/usr/bin/env sh

TARGET_DATASET=UnitTest_Images
TARGET_DB=leveldb
TARGET_IMG_TYPE=JPEG

TARGET_DIR=$HOME/Documents/Dataset/$TARGET_DATASET

# create target dir if not existing
[ ! -d $TARGET_DIR/$TARGET_DB ] && mkdir -p $TARGET_DIR/$TARGET_DB

# create the image list (txt)
find $TARGET_DIR/$TARGET_IMG_TYPE -type f -exec echo {} \; > $TARGET_DIR/image_all.txt

# transform the image list (simply with a "0" behind each image path)
sed "s/$/ 0/" $TARGET_DIR/image_all.txt > $TARGET_DIR/image_all_labeled.txt
