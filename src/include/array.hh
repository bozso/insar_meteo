#ifndef CARRAY_HH
#define CARRAY_HH

#include <stddef.h>

#include "common_macros.hh"

template <class T>
struct array {
    T* data;
    size_t size;
    
    array(): data(NULL), size(0) {};
    
    ~array() {
        Mem_Del(data);
        size = 0;
    }


    #ifndef __INMET_IMPL
    bool const init(size_t const init_size);
    bool const init(size_t const init_size, T const init_value);
    bool const init(array<T> const& original);
    array& operator= (array const & copy);
    #else

    array<T>& operator= (array<T> const& copy) {
        // Return quickly on assignment to self.
        if (this == &copy) {
            return *this;
        }
    
        // Do all operations that can generate an expception first.
        // BUT DO NOT MODIFY THE OBJECT at this stage.
        T* tmp = Mem_New(T, size);
    
        for(size_t ii = 0; ii < size; ++ii)
            tmp[ii] = copy.data[ii];
    
        // Now that you have finished all the dangerous work.
        // Do the operations that  change the object.
        //std::swap(tmp, data);
        size = copy.size;
    
        // Finally tidy up
        Mem_Del(tmp);
    
        // Now you can return
        return *this;
    }

    bool const init(size_t const init_size) {
        if ((data = Mem_New(T, init_size)) == NULL)
            return true;
        
        size = init_size;
        return false;
    }
    
    bool const init(size_t const init_size, T const init_value) {
        if ((data = Mem_New(T, init_size)) == NULL)
            return true;
    
        size = init_size;
        
        for(size_t ii = 0; ii < size; ++ii)
            data[ii] = init_value;
        
        return false;
    }
    
    bool const init(array<T> const & original) {
        size = original.size;
    
        if ((data = Mem_New(T, size)) == NULL)
            return true;
        
        for(size_t ii = 0; ii < size; ++ii)
            data[ii] = original.data[ii];
        
        return false;
    }
    
    #endif
    
    T& operator[](size_t const index) {
        return data[index];
    }
    
    
    T const operator[](size_t const index) const {
        return data[index];
    }
    
};

#endif
