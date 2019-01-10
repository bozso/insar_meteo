#ifndef VIEW_H
#define VIEW_H

#include <stdio.h>

#include "common.h"
#include "array.h"

template<typename T>
class view {
    public:
        size_t ndim, *shape, *strides;

        view(arptr const arr)
        {
            size_t datasize = arr->datasize, ndim = arr->ndim;
            
            if (arr->isnumpy)
            {
                this->isnumpy = true;
                this->strides = new size_t[ndim];
                
                for(size_t ii = ndim; ii--;)
                {
                    this->strides[ii] = size_t(double(arr->strides[ii]) / datasize);
                }
            }
            else
                this->strides = arr->strides;
                
            this->ndim = arr->ndim;
            this->shape = arr->shape;
            data = static_cast<T*>(arr->data);
        }
        
        T* get_data() const {
            return this.data;
        }

        ~view()
        {
            if (this->isnumpy)
                delete[] this->strides;
        }
        
        T& operator()(size_t ii) {
            return this->data[ii * this->strides[0]];
        }

        T& operator()(size_t ii, size_t jj) {
            return this->data[ii * this->strides[0] + jj * this->strides[1]];
        }

        T& operator()(size_t ii, size_t jj, size_t kk) {
            return this->data[ii * this->strides[0] + jj * this->strides[1] + 
                              kk * this->strides[2]];
        }

        T& operator()(size_t ii, size_t jj, size_t kk, size_t ll) {
            return this->data[ii * this->strides[0] + jj * this->strides[1] + 
                              kk * this->strides[2] + ll * this->strides[3]];
        }

        T const& operator()(size_t ii) const {
            return this->data[ii * this->strides[0]];
        }

        T const& operator()(size_t ii, size_t jj) const {
            return this->data[ii * this->strides[0] + jj * this->strides[1]];
        }

        T const& operator()(size_t ii, size_t jj, size_t kk) const {
            return this->data[ii * this->strides[0] + jj * this->strides[1] + 
                              kk * this->strides[2]];
        }

        T const& operator()(size_t ii, size_t jj, size_t kk, size_t ll) const {
            return this->data[ii * this->strides[0] + jj * this->strides[1] + 
                              kk * this->strides[2] + ll * this->strides[3]];
        }
    private:
        bool isnumpy;
        T* data;
};

#endif
