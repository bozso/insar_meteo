#ifndef __AUX_HPP
#define __AUX_HPP

#include <type_traits>
#include <complex>


#define type_assert(T, P, msg) static_assert(T<P>::value, msg)
#define m_log printf("File: %s -- Line: %d.\n", __FILE__, __LINE__)

namespace std {

template<class T>
struct is_complex : false_type {};


template<class T>
struct is_complex<complex<T>> : true_type {};

}


namespace aux {



template<class T>
using ptr = T*;

template<class T>
using cptr = ptr<T> const;


template<class T>
using ref = T&;

template<class T>
using cref = ref<T const>;


// namespace end
}

// guard
#endif