#include "capi_structs.hh"
//#include "pyvector.hh"
#include "array.hh"
#include "capi_array.hh"

#define spec_arr(type)\
template struct array<type>;\
template bool init(array<type>& arr, const size_t init_size);\
template bool init(array<type>& arr, const int init_size, const type init_value);\
template bool init(array<type>& arr, const array<type>& original);


#define spec_ndarray(type, ndim)\
template struct nparray<type, ndim>;\
template bool import(nparray<type, ndim>& arr, PyObject *__obj = NULL);\
template bool empty(nparray<type, ndim>& arr, npy_intp *dims, int fortran = 0);\
template PyArrayObject* get_array(const nparray<type, ndim>& arr);\
template PyObject* get_obj(const nparray<type, ndim>& arr);\
template const unsigned int get_shape(const nparray<type, ndim>& arr, unsigned int ii);\
template const unsigned int rows(const nparray<type, ndim>& arr);\
template const unsigned int cols(const nparray<type, ndim>& arr);\
template type* get_data(const nparray<type, ndim>& arr);


//template struct nparray<npy_double, 2>;
//template struct nparray<npy_double, 1>;
//template struct nparray<npy_bool, 1>;

//template struct vector<double>;

spec_arr(bool)

spec_ndarray(npy_double, 2)
spec_ndarray(npy_double, 1)
spec_ndarray(npy_bool, 1)
