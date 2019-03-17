#ifndef NUMPY_HPP
#define NUMPY_HPP

#include <complex>

typedef char memtype;
typedef memtype* memptr;

typedef std::complex<double> cpx128;
typedef std::complex<float> cpx64;


enum class dtype : int {
    Unknown     = 0,
    Bool        = 1,
    Int         = 2,
    Long        = 3,
    Size_t      = 4,
    Int8        = 5,
    Int16       = 6,
    Int32       = 7,
    Int64       = 8,
    UInt8       = 9,
    UInt16      = 10,
    UInt32      = 11,
    UInt64      = 12,
    Float32     = 13,
    Float64     = 14,
    Complex64   = 15,
    Complex128  = 16
};


enum layout {
    colmajor,
    rowmajor
};


template<class T1, class T2>
static inline T1 convert(memptr in)
{
    T1 ret = static_cast<T1>(*reinterpret_cast<T2*>(in));
    return ret;
}


class Array {
public:
    typedef long idx;

    int type;
    idx ndim, ndata, datasize, *shape, *strides;

    bool check_ndim(idx const ndim) const;
    bool check_type(int const type) const;
    bool check_rows(idx const rows) const;
    bool check_cols(idx const cols) const;
    
    Array() = delete;
    ~Array() = default;

    
    idx rows() const
    {
        return shape[0];
    }

    idx cols() const
    {
        return shape[1];
    }
    
    template<class T>
    T conv(idx offset)
    {
        memptr in = data + offset;

        switch(type)
        {
            case 1:
                return convert<T, bool>(in);
            case 2:
                return convert<T, int>(in);
            case 3:
                return convert<T, long>(in);
            case 4:
                return convert<T, size_t>(in);
    
            case 5:
                return convert<T, int8_t>(in);
            case 6:
                return convert<T, int16_t>(in);
            case 7:
                return convert<T, int32_t>(in);
            case 8:
                return convert<T, int64_t>(in);
    
            case 9:
                return convert<T, uint8_t>(in);
            case 10:
                return convert<T, uint16_t>(in);
            case 11:
                return convert<T, uint32_t>(in);
            case 12:
                return convert<T, uint64_t>(in);
    
            case 13:
                return convert<T, float>(in);
            case 14:
                return convert<T, float>(in);
    
            //case 15:
                //return convert<T, cpx64>(in);
            //case 16:
                //return convert<T, cpx128>(in);
        }
    }


    template<class T>
    T& get(idx ii)
    {
        return conv<T>(ii * strides[0]);
    }


    template<class T>
    T& get(idx ii, idx jj)
    {
        return conv<T>(ii * strides[0] + jj * strides[1]);
    }


    template<class T>
    T& get(idx ii, idx jj, idx kk)
    {
        return conv<T>(ii * strides[0] + jj * strides[1] + kk * strides[2]);
    }


    template<class T>
    T& get(idx ii, idx jj, idx kk, idx ll)
    {
        return conv<T>(ii * strides[0] + jj * strides[1] + kk * strides[2]
                       + kk * strides[3]);
    }


    template<class T>
    T const& get(idx ii) const
    {
        return conv<T>(ii * strides[0]);
    }

    template<class T>
    T const& get(idx ii, idx jj) const
    {
        return conv<T>(ii * strides[0] + jj * strides[1]);
    }

    template<class T>
    T const& get(idx ii, idx jj, idx kk) const
    {
        return conv<T>(ii * strides[0] + jj * strides[1] + kk * strides[2]);
    }

    
    template<class T>
    T const& get(idx ii, idx jj, idx kk, idx ll) const
    {
        return conv<T>(ii * strides[0] + jj * strides[1] + kk * strides[2]
                       + kk * strides[3]);
    }

    
    memptr get_dataptr() const {
        return data;
    }
    
private:
    memptr data;
};

typedef Array* array_ptr;



/*
void setup_view(void**data, view_meta* md, arptr const arr);

template<typename T>
class View {
    public:
        View(arptr const arr) {
            setup_view((void**) &(this->data), &(this->md), arr);
        }
        
        T* get_data() const {
            return this->data;
        }

        ~View()
        {
            if (this->md.isnumpy)
                delete[] this->md.strides;
        }
        
        size_t const ndim() const {
            return this->md.ndim;
        }
        
        size_t const shape(size_t ii) const {
            return this->md.shape[ii];
        }

    private:
        view_meta md;
        T* data;
};
*/

#endif
