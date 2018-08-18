#include <Python.h>  // NOLINT(build/include_alpha)

// Produce deprecation warnings (needs to come before arrayobject.h inclusion).
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <boost/make_shared.hpp>
#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <numpy/arrayobject.h>
#include "boost/algorithm/string.hpp"
#include "boost/pointer_cast.hpp"
#include "google/protobuf/text_format.h"

// these need to be included after boost on OS X
#include <string>  // NOLINT(build/include_order)
#include <vector>  // NOLINT(build/include_order)
#include <map>
#include <algorithm>  // NOLINT(build/include_order)

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/image_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

// Temporary solution for numpy < 1.7 versions: old macro, no promises.
// You're strongly advised to upgrade to >= 1.7.
#ifndef NPY_ARRAY_C_CONTIGUOUS
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#define PyArray_SetBaseObject(arr, x) (PyArray_BASE(arr) = (x))
#endif

// for python, we'll just use float as the type
typedef float Dtype;

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using caffe::ImageDataLayer;
using std::string;

namespace db = caffe::db;
namespace bp = boost::python;

template<typename Dtype>
bp::list feature_extraction_pipeline_raw(bp::tuple args, bp::dict kwargs) {

  if (bp::len(kwargs) > 0) {
    throw std::runtime_error("main_raw takes no kwargs");
  }

  const int num_required_args = 7;
  if (bp::len(args) < num_required_args) {
    LOG(ERROR)<<
    "This program takes in a trained network and an input data layer, and then"
    " extract features of the input data produced by the net.\n"
    "Usage: extract_features  pretrained_net_param"
    "  feature_extraction_proto_file  extract_feature_blob_name1[,name2,...]"
    "  save_feature_dataset_name1[,name2,...]  num_mini_batches  db_type"
    "  [CPU/GPU] [DEVICE_ID=0]\n"
    "Note: you can extract multiple features in one pass by specifying"
    " multiple feature blob names and dataset names separated by ','."
    " The names cannot contain white space characters and the number of blobs"
    " and datasets must be equal.";
    return bp::list();
  }

  // force to use CPU
  Caffe::set_mode(Caffe::CPU);

  std::string pretrained_binary_proto( (bp::extract<std::string>(args[1])) );

  // Expected prototxt contains at least one data layer such as
  //  the layer data_layer_name and one feature blob such as the
  //  fc7 top blob to extract features.
  /*
   layers {
     name: "data_layer_name"
     type: DATA
     data_param {
       source: "/path/to/your/images/to/extract/feature/images_leveldb"
       mean_file: "/path/to/your/image_mean.binaryproto"
       batch_size: 128
       crop_size: 227
       mirror: false
     }
     top: "data_blob_name"
     top: "label_blob_name"
   }
   layers {
     name: "drop7"
     type: DROPOUT
     dropout_param {
       dropout_ratio: 0.5
     }
     bottom: "fc7"
     top: "fc7"
   }
   */

  std::string feature_extraction_proto( (bp::extract<std::string>(args[2])) );
  boost::shared_ptr<Net<Dtype> > feature_extraction_net(
      new Net<Dtype>(feature_extraction_proto, caffe::TEST));
  feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

  std::string extract_feature_blob_names( (bp::extract<std::string>(args[3])) );
  std::vector<std::string> blob_names;
  boost::split(blob_names, extract_feature_blob_names, boost::is_any_of(","));

  std::string save_feature_dataset_names( (bp::extract<std::string>(args[4])) ) ;
  std::vector<std::string> dataset_names;
  boost::split(dataset_names, save_feature_dataset_names,
               boost::is_any_of(","));
  size_t num_features = blob_names.size();


  int num_mini_batches = bp::extract<int>(args[5]);

  std::vector<boost::shared_ptr<db::DB> > feature_dbs;
  std::vector<boost::shared_ptr<db::Transaction> > txns;
  const char* db_type = bp::extract<const char*>(args[6]);
  for (size_t i = 0; i < dataset_names.size(); ++i) {
    //LOG(INFO)<< "Opening dataset " << dataset_names[i];
    boost::shared_ptr<db::DB> db(db::GetDB(db_type));
    db->Open(dataset_names.at(i), db::WRITE);
    feature_dbs.push_back(db);
    boost::shared_ptr<db::Transaction> txn(db->NewTransaction());
    txns.push_back(txn);
  }

  // storing "fc7" features (in C++)
  Datum datum; 
  // sorting "prob" features (in Python)
  bp::list prob_list;

  // the name of the image as the 'key' for feature db
  const boost::shared_ptr<ImageDataLayer<Dtype> > layer_ptr = boost::static_pointer_cast<ImageDataLayer<Dtype> >(feature_extraction_net->layer_by_name("data"));
  const std::vector<std::pair<std::string, int> >& data_names_labels = layer_ptr->GetDataNameLabel();

  std::vector<int> image_indices(num_features, 0);
  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
    feature_extraction_net->Forward();
    for (int i = 0; i < num_features; ++i) {
      const boost::shared_ptr<Blob<Dtype> > feature_blob =
        feature_extraction_net->blob_by_name(blob_names[i]);
      int batch_size = feature_blob->num();
      int dim_features = feature_blob->count() / batch_size;
      const Dtype* feature_blob_data;
      // blob "fc7"
      if(i == 0){
        for (int n = 0; n < batch_size; ++n) {
          datum.set_height(feature_blob->height());
          datum.set_width(feature_blob->width());
          datum.set_channels(feature_blob->channels());
          datum.clear_data();
          datum.clear_float_data();
          feature_blob_data = feature_blob->cpu_data() +
              feature_blob->offset(n);
          for (int d = 0; d < dim_features; ++d) {
            datum.add_float_data(feature_blob_data[d]);
          }
          string key_str = data_names_labels[batch_index].first;

          // const string ready to be stored
          string out;
          datum.SerializeToString(&out);

          txns.at(i)->Put(key_str, out);
          ++image_indices[i];
          if (image_indices[i] % 1000 == 0) {
            txns.at(i)->Commit();
            txns.at(i).reset(feature_dbs.at(i)->NewTransaction());
          }
        } // for (int n = 0; n < batch_size; ++n)
      } // if(i == 0)
      // blob "prob"
      else{
        for (int n = 0; n < batch_size; ++n) {
          feature_blob_data = feature_blob->cpu_data() +
              feature_blob->offset(n);
          for (int d = 0; d < dim_features; ++d) {
            prob_list.append(feature_blob_data[d]);
            //datum.add_float_data(feature_blob_data[d]);
          }
        } // for (int n = 0; n < batch_size; ++n)
      }
    }  // for (int i = 0; i < num_features; ++i)
  }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
  // write the last batch
  for (int i = 0; i < dataset_names.size(); ++i) {
    if (image_indices[i] % 1000 != 0) {
      txns.at(i)->Commit();
    }
    feature_dbs.at(i)->Close();
  }

  printf("%s\n", "Successfully extracted the features!");
  return prob_list;
}

// costum compare object 
template <typename K, typename V>
bool cmp(const std::pair<K, V>& lhs, const std::pair<K,V>& rhs)
{
  return lhs.second > rhs.second;
}

// using STL pair for argSort                                                                                                                                      
template <typename Dtype>
std::vector<size_t> argsort_using_stlMap(std::vector<Dtype> const& values) {
  std::vector<size_t> indices(values.size());
  std::vector<std::pair<size_t, Dtype> > pairs;
  for(size_t i=0; i<values.size(); ++i){
    pairs.push_back(std::make_pair(i, values[i]));
  }

  std::sort(pairs.begin(), pairs.end(), cmp<size_t, Dtype> );

  for(size_t i=0; i<pairs.size(); ++i){
    indices[i] = pairs[i].first;
  }
  return indices;
}

// Computing the similarity of two features (e.g. *Cosine similarity or L2-norm)
template <typename Dtype>
float similarity(Datum& datum_1, Datum& datum_2){

  int feature_dim = datum_1.channels();
  Dtype* ptr_1 = datum_1.mutable_float_data()->mutable_data();
  Dtype* ptr_2 = datum_2.mutable_float_data()->mutable_data();

  Dtype dot_val, norm_1, norm_2;

  // L2-norm
  norm_1 = sqrt( caffe::caffe_cpu_dot(feature_dim, ptr_1, ptr_1) );
  norm_2 = sqrt( caffe::caffe_cpu_dot(feature_dim, ptr_2, ptr_2) );
  // dot product
  dot_val = caffe::caffe_cpu_dot(feature_dim, ptr_1, ptr_2);

  return dot_val / (norm_1 * norm_2); 
}

/* DEBUG -> dislay Datum instance @Wei-Lin */
template <typename Dtype>
void listDatum(const Datum& datum){
  if(datum.has_height())  std::cout << "H:\t" << datum.height() << std::endl;
  if(datum.has_width())  std::cout << "W:\t" << datum.width() << std::endl;
  if(datum.has_channels())  std::cout << "C:\t" << datum.channels() << std::endl;
  std::cout << "size of float data:\t" << datum.float_data_size() << std::endl;

  // display fc7 floating features
  google::protobuf::RepeatedField<Dtype> data = datum.float_data();
  for( google::protobuf::RepeatedField<float>::iterator iter=data.begin(); iter<data.end(); ++iter ){
    std::cout << *iter << " ";
  }
  std::cout << "\n" << std::endl;
}

template <typename Dtype>
bp::object matching_pipeline_raw(bp::tuple args, bp::dict kwargs) {

  if (bp::len(kwargs) > 0) {
    throw std::runtime_error("main_raw takes no kwargs");
  }

  const int num_required_args = 3;
  if (bp::len(args) < num_required_args) {
    LOG(ERROR)<<
    "This program takes in an image feature database (stored in leveldb/lmdb format)" 
    "and a query image feature, and then"
    " compare database features with the query feature to give a short list of relavence.\n"

    "@TODO: define the usage of retrieval interface"
    
    "Usage: retrieval path_to_feature_db db_type\n"

    "@TODO: modify the notes for retrieval process"

    "Note: you can extract multiple features in one pass by specifying"
    " multiple feature blob names and dataset names separated by ','."
    " The names cannot contain white space characters and the number of blobs"
    " and datasets must be equal.";
    return bp::object();
  }
 
  // 1st argument: the path to feature db
  std::string feature_db_name( (bp::extract<std::string>(args[1])) );
  std::cout << "feature_db_name: " << feature_db_name << std::endl;
  // 2nd argument: the type of the feature db
  const char* db_type = bp::extract<const char*>(args[2]);

  // Open the feature db
  boost::shared_ptr<db::DB> feature_db(db::GetDB(db_type));
  feature_db->Open(feature_db_name, db::READ);

  // Visit all the features
  std::vector<boost::shared_ptr<Datum> > Datum_vec;
  db::Cursor* db_cursor(feature_db->NewCursor());
  for(db_cursor->SeekToFirst(); db_cursor->valid(); db_cursor->Next()){
    // get the current string value
    std::string s = db_cursor->value();

    // convert to the feature (floating)
    boost::shared_ptr<Datum> datum(new Datum);
    datum->ParseFromString(s);

    Datum_vec.push_back(datum);
  }
  printf("Total feat. is #%zd\n", Datum_vec.size());

  // similarity measure of two features
  Dtype dot_val = similarity<Dtype>(*Datum_vec[0], *Datum_vec[1]);
  std::cout << "The similarity of vector[0] and vector[1] is " << dot_val << std::endl;

  // close the feature db
  feature_db->Close();


  return bp::object();
}

template<typename Dtype>
bp::list retrieval_pipeline_raw(bp::tuple args, bp::dict kwargs) {

  if (bp::len(kwargs) > 0) {
    throw std::runtime_error("main_raw takes no kwargs");
  }

  const int num_required_args = 7;
  if (bp::len(args) < num_required_args) {
    LOG(ERROR)<<
    "This program takes in a trained network and an input data layer, and then"
    " extract features of the input data produced by the net.\n"
    "Usage: extract_features  pretrained_net_param"
    "  feature_extraction_proto_file  extract_feature_blob_name1[,name2,...]"
    "  save_feature_dataset_name1[,name2,...]  num_mini_batches  db_type"
    "  [CPU/GPU] [DEVICE_ID=0]\n"
    "Note: you can extract multiple features in one pass by specifying"
    " multiple feature blob names and dataset names separated by ','."
    " The names cannot contain white space characters and the number of blobs"
    " and datasets must be equal.";
    return bp::list();
  }

  // Expected prototxt contains at least one data layer such as
  //  the layer data_layer_name and one feature blob such as the
  //  fc7 top blob to extract features.
  /*
   layers {
     name: "data_layer_name"
     type: DATA
     data_param {
       source: "/path/to/your/images/to/extract/feature/images_leveldb"
       mean_file: "/path/to/your/image_mean.binaryproto"
       batch_size: 128
       crop_size: 227
       mirror: false
     }
     top: "data_blob_name"
     top: "label_blob_name"
   }
   layers {
     name: "drop7"
     type: DROPOUT
     dropout_param {
       dropout_ratio: 0.5
     }
     bottom: "fc7"
     top: "fc7"
   }
   */

  // force caffe to use CPU
  Caffe::set_mode(Caffe::CPU);

  // 1st argument: excutable binary name (ingored)

  // 2nd argument: model binary
  std::string pretrained_binary_proto( (bp::extract<std::string>(args[1])) );

  // 3rd argument: model proto txt
  std::string feature_extraction_proto( (bp::extract<std::string>(args[2])) );
  boost::shared_ptr<Net<Dtype> > feature_extraction_net(
      new Net<Dtype>(feature_extraction_proto, caffe::TEST));
  feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

  // 4nd argument: blob names (features to be extracted)
  std::string extract_feature_blob_names( (bp::extract<std::string>(args[3])) );
  std::vector<std::string> blob_names;
  boost::split(blob_names, extract_feature_blob_names, boost::is_any_of(","));
  size_t num_features = blob_names.size();

  // 5nd argument: size of mini patch
  int num_mini_batches = bp::extract<int>(args[4]);

  // 6nd argument: feature db path
  std::string feature_db_name( (bp::extract<std::string>(args[5])) );

  // 7nd argument: feature db type
  const char* db_type = bp::extract<const char*>(args[6]);

  // storing "fc7" features (in C++)
  Datum query;
  // storing sorted "filenames" (in Python)
  bp::list result_list; 
  // storing "prob" features (in Python)
  bp::list prob_list;

  // the name of the image as the 'key' for feature db
  const boost::shared_ptr<ImageDataLayer<Dtype> > layer_ptr = boost::static_pointer_cast<ImageDataLayer<Dtype> >(feature_extraction_net->layer_by_name("data"));
  const std::vector<std::pair<std::string, int> >& data_names_labels = layer_ptr->GetDataNameLabel();

  /* ==== Feature Extraction ==== */
  std::vector<int> image_indices(num_features, 0);
  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
    feature_extraction_net->Forward();
    for (int i = 0; i < num_features; ++i) {
      const boost::shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net->blob_by_name(blob_names[i]);
      int batch_size = feature_blob->num();
      int dim_features = feature_blob->count() / batch_size;
      const Dtype* feature_blob_data;
      // blob[fc7]
      if(i == 0){
        for (int n = 0; n < batch_size; ++n) {
          query.set_height(feature_blob->height());
          query.set_width(feature_blob->width());
          query.set_channels(feature_blob->channels());
          query.clear_data();
          query.clear_float_data();
          feature_blob_data = feature_blob->cpu_data() + feature_blob->offset(n);
          for (int d = 0; d < dim_features; ++d) {
            query.add_float_data(feature_blob_data[d]);
          }

          // const string ready to be stored
          string out;
          query.SerializeToString(&out);
        }
      }
      // blob[prob]
      else{
        for (int n = 0; n < batch_size; ++n) {
          feature_blob_data = feature_blob->cpu_data() +
              feature_blob->offset(n);
          for (int d = 0; d < dim_features; ++d) {
            prob_list.append(feature_blob_data[d]);
            //datum.add_float_data(feature_blob_data[d]);
          }
        } // for (int n = 0; n < batch_size; ++n)
      }
    }  // for (int i = 0; i < num_features; ++i)
  }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)

  /* ==== Retrieval ==== */
  // Open the feature db
  boost::shared_ptr<db::DB> feature_db(db::GetDB(db_type));
  feature_db->Open(feature_db_name, db::READ);
  db::Cursor* db_cursor(feature_db->NewCursor());
  
  // Visit all the features
  std::vector<std::string> name_vec;
  std::vector<boost::shared_ptr<Datum> > Datum_vec;
  for(db_cursor->SeekToFirst(); db_cursor->valid(); db_cursor->Next()){
    std::string name = db_cursor->key();
    // get the current string value
    std::string feat = db_cursor->value();

    // convert to the feature (floating)
    boost::shared_ptr<Datum> datum(new Datum);
    datum->ParseFromString(feat);

    name_vec.push_back(name);
    Datum_vec.push_back(datum);
  }
  printf("#feature in DB = #%zd\n", Datum_vec.size());

  // similarity measure of features against query
  std::vector<Dtype> scores;
  for(std::vector< boost::shared_ptr<Datum> >::iterator iter=Datum_vec.begin(); iter<Datum_vec.end(); ++iter){
    Dtype dot_val = similarity<Dtype>(query, **iter);
    scores.push_back(dot_val);
    std::cout << "The similarity of two vectors is " << dot_val << std::endl;
  }

  // sort by similarity measures
  std::vector<size_t> indices = argsort_using_stlMap(scores);
  for(std::vector<size_t>::iterator iter=indices.begin(); iter<indices.end(); ++iter){
    std::cout << "Sorted Indices: " << *iter << std::endl;
  }

  // map sorted filename into python list
  for (size_t i=0; i<indices.size(); ++i) {
    result_list.append(name_vec[ indices[i] ]);
  }

  // close the feature db
  feature_db->Close();

  printf("%s\n", "Successfully retrieved similar images!");
  return result_list;
}


BOOST_PYTHON_MODULE(_django_CV){
  bp::def("feature_extraction_pipeline", bp::raw_function(&feature_extraction_pipeline_raw<Dtype>));
  bp::def("matching_pipeline", bp::raw_function(&matching_pipeline_raw<Dtype>));
  bp::def("retrieval_pipeline", bp::raw_function(&retrieval_pipeline_raw<Dtype>));
  // boost python expects a void (missing) return value, while import_array
  // returns NULL for python3. import_array1() forces a void return value.
  import_array1();
}
