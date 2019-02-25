#ifndef STL_INST_HPP
#define STL_INST_HPP

#include <vector>
#include <memory>

#include <ltl/marray.h>

extern template class std::vector<double>;
extern template class ltl::MArray<double,2>;

#endif
