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
#include "google/protobuf/repeated_field.h"

// these need to be included after boost on OS X
#include <string>  // NOLINT(build/include_order)
#include <vector>  // NOLINT(build/include_order)

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using std::string;
namespace db = caffe::db;

// for python, we use float as Dtype
typedef float Dtype;

// Computing the similarity of two features (e.g. *Cosine similarity or L2-norm)
template<typename Dtype>
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

/* DEBUG -> diaply Datum instance @Wei-Lin */
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

template<typename Dtype>
bp::object retrieval_pipeline_raw(bp::tuple args, bp::dict kwargs) {

  if (bp::len(kwargs) > 0) {
    throw std::runtime_error("main_raw takes no kwargs");
  }

  const int num_required_args = 3;
  if (argc < num_required_args) {
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
  int arg_pos = num_required_args;

  arg_pos = 0;  
  // 1st argument: the path to feature db
  std::string feature_db_name(bp::extract<std::string>(args[1]));
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

BOOST_PYTHON_MODULE(_retrieval){
  bp::def("retrieval_pipeline", bp::raw_function(&retrieval_pipeline_raw<Dtype>));
  // boost python expects a void (missing) return value, while import_array
  // returns NULL for python3. import_array1() forces a void return value.
  import_array1();
}
