#ifndef ARRAY_H
#define ARRAY_H

/******************************
 * Wrapper object for arrays. *
 ******************************/

template<typename T, unsigned int ndim>
struct array {
    unsigned int shape[ndim], strides[ndim];
    T * data;
    
    array() {};
    
    array(T * _data, int * _shape) {
        unsigned int shape_sum = 0;
    
        for(unsigned int ii = 0; ii < ndim; ++ii)
            shape[ii] = static_cast<unsigned int>(_shape[ii]);
        
        for(unsigned int ii = 0; ii < ndim; ++ii) {
            shape_sum = 1;
            
            for(unsigned int jj = ii + 1; jj < ndim; ++jj)
                 shape_sum *= shape[jj];
            
            strides[ii] = shape_sum;
        }
        data = _data;
    }
    
    unsigned int get_shape(unsigned int ii)
    {
        return shape[ii];
    }

    unsigned int get_rows()
    {
        return shape[0];
    }
    
    unsigned int get_cols()
    {
        return shape[1];
    }
    
    T& operator()(unsigned int ii)
    {
        return data[ii * strides[0]];
    }

    T& operator()(unsigned int ii, unsigned int jj)
    {
        return data[ii * strides[0] + jj * strides[1]];
    }
    
    T& operator()(unsigned int ii, unsigned int jj, unsigned int kk)
    {
        return data[ii * strides[0] + jj * strides[1] + kk * strides[2]];
    }

    T& operator()(unsigned int ii, unsigned int jj, unsigned int kk,
                  unsigned int ll)
    {
        return data[  ii * strides[0] + jj * strides[1] + kk * strides[2]
                    + ll * strides[3]];
    }
};

#endif
