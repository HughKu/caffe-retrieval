#include <Python.h>  // NOLINT(build/include_alpha)

// Produce deprecation warnings (needs to come before arrayobject.h inclusion).
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <boost/make_shared.hpp>
#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <numpy/arrayobject.h>

// these need to be included after boost on OS X
#include <string>  // NOLINT(build/include_order)
#include <vector>  // NOLINT(build/include_order)

// Temporary solution for numpy < 1.7 versions: old macro, no promises.
// You're strongly advised to upgrade to >= 1.7.
#ifndef NPY_ARRAY_C_CONTIGUOUS
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#define PyArray_SetBaseObject(arr, x) (PyArray_BASE(arr) = (x))
#endif

namespace bp = boost::python;

bp::object main_raw(bp::tuple args, bp::dict kwargs){
  if (bp::len(kwargs) > 0) {
    throw std::runtime_error("main_raw takes no kwargs");
  }
  for (int i=0; i<bp::len(args); ++i) {
    std::string arg = bp::extract<std::string>(args[i]);
    printf("arg-%d is %s\n", i, arg.c_str());
  }
  // We need to explicitly return None to use bp::raw_function.
  return bp::object();
}

template <typename Dtype>
class World {
  public:
    World(std::string msg, Dtype value): msg(msg), value(value) { printf("%s/%f is initialized.\n", this->msg.c_str(), this->value); }
    // string property
    void setString(std::string msg) { this->msg = msg; }
    std::string getString() { return this->msg; }
    // value property
    void setValue(int value) { this->value = value; }
    int getValue() {return this->value; }

  private:
    // class members
    std::string msg;
    Dtype value;
};

typedef float Dtype;
BOOST_PYTHON_MODULE(_bp_test) {
  bp::class_<World<Dtype> >("World", bp::init<std::string, Dtype>())
    .add_property("msg", &World<Dtype>::getString, &World<Dtype>::setString)
    .add_property("val", &World<Dtype>::getValue, &World<Dtype>::setValue)
  ;

  bp::def("main", bp::raw_function(&main_raw));
  
  // boost python expects a void (missing) return value, while import_array
  // returns NULL for python3. import_array1() forces a void return value.
  import_array1();
}


