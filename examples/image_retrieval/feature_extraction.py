from py_caffe_module import _extract_features

# Python Code (this file)
BIN_NAME = __file__
# Parent directory of dataset (fixed)
PARENT_DB = "/Users/wlku/Documents/Dataset"
# Blob Name
BLOB_NAME = "fc7"
# Num of Image to be processed
NUM_IMAGES = 4

######################
# == Album-related ==
######################
ALBUM_DATASET = "UnitTest_Images"
ALBUM_FEAT_TYPE = "leveldb"
ALBUM_IMG_TYPE = "JPEG"
ALBUM_DIR = "{0}/{1}".format(PARENT_DB, ALBUM_DATASET)
ALBUM_FEATURE_NAME = "{0}/{1}/feature_{2}".format(ALBUM_DIR, ALBUM_FEAT_TYPE, BLOB_NAME)

######################
# == Model-related ==
######################
MODEL_BIN_PATH = "../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
MODEL_PROTO_PATH = "{0}/imagenet_val.prototxt".format(ALBUM_DIR)

# Entry point
main = _extract_features.feature_extraction_pipeline

# Run Feature Extraction
main(BIN_NAME, MODEL_BIN_PATH, MODEL_PROTO_PATH, BLOB_NAME, ALBUM_FEATURE_NAME, NUM_IMAGES, ALBUM_FEAT_TYPE)

