#include "capi_macros.hh"
#include "capi_structs.hh"


template <class T>
T& array<T>::operator[](size_t index)
{
    return data[index];
}


template <class T>
T const& Array<T>::operator[] const (size_t index)
{
    return data[index];
}


template <class T>
array<T>& array<T>::operator= (const array& copy)
{
    // Return quickly on assignment to self.
    if (this == &copy) {
        return *this;
    }

    // Do all operations that can generate an expception first.
    // BUT DO NOT MODIFY THE OBJECT at this stage.
    T* tmp = Mem_New(T, size);
    
    FOR(ii, 0, size) {
        tmp[ii] = copy.data[ii];
    }

    // Now that you have finished all the dangerous work.
    // Do the operations that  change the object.
    std::swap(tmp, data);
    size = copy.size;

    // Finally tidy up
    Mem_Del(tmp);

    // Now you can return
    return *this;
}


template <class T>
array<T>::~array()
{
    Mem_Del(data);
    size = 0;
}


template <class T>
bool init(array<T>& arr, const size_t init_size)
{
    if ((arr.data = Mem_New(T, init_size)) == NULL)
        return true;
    
    arr.size = init_size;
    return false;
}


template <class T>
bool init(array<T>& arr, const int init_size, const T init_value)
{
    if ((arr.data = Mem_New(T, init_size)) == NULL)
        return true;

    arr.size = init_size;
    
    FOR(ii, 0, size) {
        arr.data[ii] = init_value;
    }
    return false;
}


template <class T>
bool init(array<T>& arr, const array& original)
{
    arr.size = original.size;

    if ((arr.data = Mem_New(T, init_size)) == NULL)
        return true;
    
    FOR(ii, 0, size) {
        arr.data[ii] = original.arr[ii];
    }
}
