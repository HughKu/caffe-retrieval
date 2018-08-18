import os
import time
import cPickle
import numpy as np
import pandas as pd
from py_caffe_module import _django_CV

class FeatureExtraction(object):

	def __init__(self):


		self.REPO_DIRNAME = os.path.abspath(os.path.dirname(__file__) + '/../..')
		# Python Code (this file)
		self.BIN_NAME = __file__
		# Parent directory of dataset (fixed)
		self.PARENT_DB = "/Users/wlku/Documents/Dataset"
		# Blob Name
		self.BLOB_NAME_FEAT = "fc7"
		self.BLOB_NAME_PROB = "prob"
		# Num of Image to be processed
		self.NUM_IMAGES = 1

		######################
		# == Album-related ==
		######################
		self.ALBUM_DATASET = "UnitTest_Django"
		self.ALBUM_FEAT_TYPE = "leveldb"
		self.ALBUM_IMG_TYPE = "JPEG"
		self.ALBUM_DIR = "{0}/{1}".format(self.PARENT_DB, self.ALBUM_DATASET)
		self.ALBUM_FEATURE_NAME = "{0}/{1}/feature_{2}".format(self.ALBUM_DIR, self.ALBUM_FEAT_TYPE, self.BLOB_NAME_FEAT)

		######################
		# == Model-related ==
		######################
		self.MODEL_BIN_PATH = "{0}/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel".format(self.REPO_DIRNAME)
		self.BET_FILE 		= "{0}/data/ilsvrc12/imagenet.bet.pickle".format(self.REPO_DIRNAME)
		self.CLASS_LABEL_FILE = "{0}/data/ilsvrc12/synset_words.txt".format(self.REPO_DIRNAME)
		self.MODEL_PROTO_PATH = "{0}/imagenet_val.prototxt".format(self.ALBUM_DIR)
		self.IMAGE_PROTO_PATH = "{0}/image_all_labeled.txt".format(self.ALBUM_DIR)

		###########################
		# == Class-label related == 
		###########################
		with open(self.CLASS_LABEL_FILE) as f:
			labels_df = pd.DataFrame([
				{
					'synset_id': l.strip().split(' ')[0],
					'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
				}
				for l in f.readlines()
			])
		self.LABELS = labels_df.sort('synset_id')['name'].values
		self.BET = cPickle.load(open(self.BET_FILE))

		# A bias to prefer children nodes in single-chain paths
		# I am setting the value to 0.1 as a quick, simple model.
		# We could use better psychological models here...
		self.BET['infogain'] -= np.array(self.BET['preferences']) * 0.1


	def extract(self, filename):
		# Write to the labeled image list
		fout = open(self.IMAGE_PROTO_PATH, 'w')
		fout.write("{0} 0".format(os.path.abspath(filename)))
		fout.close()

		# Entry point
		main = _django_CV.feature_extraction_pipeline
		# Run Feature Extraction
		starttime = time.time()
		scores = main(self.BIN_NAME, self.MODEL_BIN_PATH, self.MODEL_PROTO_PATH, "{0},{1}".format(self.BLOB_NAME_FEAT, self.BLOB_NAME_PROB), 
			self.ALBUM_FEATURE_NAME, self.NUM_IMAGES, self.ALBUM_FEAT_TYPE)
		endtime = time.time()

		scores = np.asarray(scores); # python list to np.array
		indices = (-scores).argsort()[:5]
		predictions = self.LABELS[indices]

		# Score:Class-label Mapping

		# In addition to the prediction text, we will also produce
		# the length for the progress bar visualization.
		meta = [
			(p, '%.5f' % scores[i])
			for i, p in zip(indices, predictions)
		]
		#print "result: {0}".format(str(meta))

		# Compute expected information gain
		expected_infogain = np.dot(self.BET['probmat'], scores[self.BET['idmapping']])
		expected_infogain *= self.BET['infogain']

		# sort the scores
		infogain_sort = expected_infogain.argsort()[::-1]
		bet_result = [(self.BET['words'][v], '%.5f' % expected_infogain[v]) for v in infogain_sort[:5]]
		#print "bet result: {0}".format(str(bet_result))

		return (True, meta, bet_result, '%.3f' % (endtime - starttime))

class ImageMatching(object):

	def __init__(self):


		self.REPO_DIRNAME = os.path.abspath(os.path.dirname(__file__) + '/../..')
		# Python Code (this file)
		self.BIN_NAME = __file__
		# Parent directory of dataset (fixed)
		self.PARENT_DB = "/Users/wlku/Documents/Dataset"
		# Blob Name
		self.BLOB_NAME_FEAT = "fc7"

		######################
		# == Album-related ==
		######################
		self.ALBUM_DATASET = "UnitTest_Django"
		self.ALBUM_FEAT_TYPE = "leveldb"
		self.ALBUM_IMG_TYPE = "JPEG"

	def match(self):

		# Entry point
		main = _django_CV.matching_pipeline
		# Run Feature Extraction
		starttime = time.time()
		scores = main(self.BIN_NAME, "{0}/{1}/{2}/feature_{3}".format(self.PARENT_DB, self.ALBUM_DATASET, self.ALBUM_FEAT_TYPE, self.BLOB_NAME_FEAT), self.ALBUM_FEAT_TYPE)
		endtime = time.time()

class ImageRetrieval(object):

	def __init__(self):


		self.REPO_DIRNAME = os.path.abspath(os.path.dirname(__file__) + '/../..')
		# Python Code (this file)
		self.BIN_NAME = __file__
		# Parent directory of dataset (fixed)
		self.PARENT_DB = "/Users/wlku/Documents/Dataset"
		# Blob Name
		self.BLOB_NAME_FEAT = "fc7"
		self.BLOB_NAME_PROB = "prob"
		# Num of Image to be processed
		self.NUM_IMAGES = 1

		######################
		# == Album-related ==
		######################
		self.ALBUM_DATASET = "UnitTest_Django"
		self.ALBUM_FEAT_TYPE = "leveldb"
		self.ALBUM_IMG_TYPE = "JPEG"
		self.ALBUM_DIR = "{0}/{1}".format(self.PARENT_DB, self.ALBUM_DATASET)
		self.ALBUM_FEATURE_NAME = "{0}/{1}/feature_{2}".format(self.ALBUM_DIR, self.ALBUM_FEAT_TYPE, self.BLOB_NAME_FEAT)

		######################
		# == Model-related ==
		######################
		self.MODEL_BIN_PATH = "{0}/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel".format(self.REPO_DIRNAME)
		self.BET_FILE 		= "{0}/data/ilsvrc12/imagenet.bet.pickle".format(self.REPO_DIRNAME)
		self.CLASS_LABEL_FILE = "{0}/data/ilsvrc12/synset_words.txt".format(self.REPO_DIRNAME)
		self.MODEL_PROTO_PATH = "{0}/imagenet_val.prototxt".format(self.ALBUM_DIR)
		self.IMAGE_PROTO_PATH = "{0}/image_all_labeled.txt".format(self.ALBUM_DIR)

		###########################
		# == Class-label related == 
		###########################
		with open(self.CLASS_LABEL_FILE) as f:
			labels_df = pd.DataFrame([
				{
					'synset_id': l.strip().split(' ')[0],
					'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
				}
				for l in f.readlines()
			])
		self.LABELS = labels_df.sort('synset_id')['name'].values
		self.BET = cPickle.load(open(self.BET_FILE))

		# A bias to prefer children nodes in single-chain paths
		# I am setting the value to 0.1 as a quick, simple model.
		# We could use better psychological models here...
		self.BET['infogain'] -= np.array(self.BET['preferences']) * 0.1

		print "Retreival Construct Initialized"


	def retrieve(self, filename):

		print "Ready to write {0} into text.".format(os.path.abspath(filename))
		# Write to the labeled image list
		fout = open(self.IMAGE_PROTO_PATH, 'w')
		fout.write("{0} 0".format(os.path.abspath(filename)))
		fout.close()

		print "Write {0} into text.".format(os.path.abspath(filename))

		# Entry point
		main = _django_CV.retrieval_pipeline
		# Run Feature Extraction
		starttime = time.time()
		filename_list = main(self.BIN_NAME, self.MODEL_BIN_PATH, self.MODEL_PROTO_PATH, "{0},{1}".format(self.BLOB_NAME_FEAT, self.BLOB_NAME_PROB), 
			self.NUM_IMAGES, self.ALBUM_FEATURE_NAME, self.ALBUM_FEAT_TYPE)
		endtime = time.time()


		return filename_list