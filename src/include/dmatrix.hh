#ifndef DMATRIX_HH
#define DMATRIX_HH

#include <stdlib.h> // size_t, malloc(), free(), realloc()
#include <string.h> // memset(), memcpy(), memmove()

#include "common_macros.hh"

#define __unpack (void**) &data, md


namespace dmat_impl {

static const size_t size_t_msb = ((size_t)1) << (sizeof(size_t)*8 - 1);
static const size_t size_t_all_but_msb = (((size_t)1) << (sizeof(size_t)*8 - 1))-1;

INMET_INLINE void
init(void** p, dmat_meta& meta, void const* buf, size_t const buf_cap);

INMET_INLINE bool
maybegrow(void** arr, dmat_meta& md, size_t const min_needed);

INMET_INLINE bool
maybegrowadd(void** arr, dmat_meta& md, size_t const num_add);

INMET_DEF void
free(void** p, dmat_meta& md);

INMET_DEF void
shrink_to_fit(void** arr, dmat_meta& md);

INMET_INLINE bool
insert(void** arr, dmat_meta& md, size_t idx, size_t n, bool const init0);

INMET_INLINE bool
add(void** arr, dmat_meta& md, size_t n, bool const init0);

INMET_INLINE void
del(void** arr, dmat_meta& md, size_t const idx, size_t const n)

INMET_INLINE void
delfast(void** arr, dmat_meta& md, size_t const idx, size_t const n)

INMET_INLINE size_t
idx(dmat_meta const& md, size_t const ii);
}

struct dmat_meta {
    size_t cnt, cap, itemsize, ncol;
};


template<class T>
struct dmatrix {
    T* data;
    dmat_meta md;

    dmatrix(): data(NULL), md.cnt(0), md.cap(0), md.ncol(0), md.itemsize(sizeof(T)) {};
    
    dmatrix(T* buf, size_t const buf_cap, size_t const ncol):
    md.ncol(ncol), md.itemsize(sizeof(T)) {
        dmat_impl::init(__unpack, buf, buf_cap);
    }

    //bool init(size_t const buf_cap, size_t const ncol)
    //{
        //_init(__unpack, 
    //}

    ~dmatrix() { dmat_impl::free(__unpack); }

    // returns a pointer to the first element of the array
    // (together with INMET_end() you can do C++-style iterating)
    T* begin() const { return data; }

    // returns a pointer to the past-the-end element of the array
    // Allows C++-style iterating, in case you're into that kind of thing:
    // for(T *it=a.begin(), *end = a.end(); it != end; ++it) foo(*it);
    // (see a.lastptr() to get a pointer *to* the last element)
    T* end() const { return data + md.cnt; }
    
    // returns (logical) number of elements currently in the array
    size_t count() const { return md.cnt; }

    // get the current reserved capacity of the array
    size_t capacity() const { return md.cap & dmat_impl::size_t_all_but_msb; }
    
    // returns true if the array is empty, else false
    bool empty() const { return md.cnt == 0; }
    

    T& operator[](size_t ii) { return data[ii]; }

    T const operator[](size_t ii) const { return data[ii]; }

    
    bool const push(T const& elem)
    {
        return dmat_impl::maybegrowadd(__unpack, 1) ?
               ((data[md.cnt++] = elem), false) : false;
    }
    

    void addn(T* vals, size_t const n)
    {
        INMET_ASSERT(vals != NULL, "Don't pass NULL vals vals to addn!");
        if (vals != NULL && dmat_impl::add(__unpack, n, false))
            size_t i_ = md.cnt - n, v_ = 0;
        
        while(i_ < md.cnt)
            data[ii_++] = vals[v_++];
	}

    void addrow(T* vals)
    {
        INMET_ASSERT(vals != NULL, "Don't pass NULL vals vals to addn!");
        
        if (vals != NULL && dmat_impl::add(__unpack, md.ncol, false))
            size_t i_ = md.cnt - n, v_ = 0;
        
        while(i_ < md.ncol)
            data[ii_++] = vals[v_++];
	}

    
    // add n elements to the end of the array and zeroe them with memset()
    // returns pointer to first added element, NULL if out of memory (array is empty then)
    T* addn(size_t const n, bool const zeroed = false)
    {
        return dmat_impl::add(__unpack, n, zeroed) ? &(data[md.cnt - n]) : NULL;
    }


    // removes all elements from the array, but does not free the buffer
    // (if you want to free the buffer too, just use free())
    void clear() { md.cnt = 0 }


    // sets the logical number of elements in the array
    // if cnt > INMET_count(a), the logical count will be increased accordingly
    // and the new elements will be uninitialized
    bool const setcount(size_t const n)
    {
        dmat_impl::maybegrow(__unpack, n) ? (md.cnt = n) : false;
    }


    // make sure the array can store cap elements without reallocating
    // logical count remains unchanged
    bool const reserve(size_t const cap)
    {
        return dmat_impl::maybegrow(__unpack, cap);
    }


    // this makes sure a only uses as much memory as for its elements
    // => maybe useful if a used to contain a huge amount of elements,
    //    but you deleted most of them and want to free some memory
    // Note however that this implies an allocation and copying the remaining
    // elements, so only do this if it frees enough memory to be worthwhile!
    void shrink_to_fit()
    {
        dmat_impl::shrink_to_fit(__unpack);
    }

    // returns the pointer *to* the last element of the array
    // (in contrast to INMET_end() which returns a pointer *after* the last element)
    // returns NULL if array is empty
    T* lastptr() const { return md.cnt > 0 ? (data + md.cnt - 1) : NULL; }


    // get pointer to element at index idx (like &a.p[idx]), but with checks
    // (unless you disabled them with #define INMET_INDEX_CHECK_LEVEL 0)
    // if index-checks are disabled, it returns NULL on invalid index (else it asserts() before returning)
    T* getptr(size_t const idx) const
    {
        return checkidx(md, idx), (idx < md.cnt) ? (data + idx) : NULL;
    }


    // returns 1 if the last (re)allocation when inserting failed (Out Of Memory)
    //   or if the array has never allocated any memory yet, else 0
    // deleting the contents when growing fails instead of keeping old may seem
    // a bit uncool, but it's simple and OOM should rarely happen on modern systems
    // anyway - after all you need to deplete both RAM and swap/pagefile.sys
    // or deplete the address space, which /might/ happen with 32bit applications
    // but probably not with 64bit (at least in the foreseeable future)
    bool oom() const { return md.cap == 0; }
};



#if (INMET_INDEX_CHECK_LEVEL == 1) || (INMET_INDEX_CHECK_LEVEL == 3)

	// removes and returns the last element of the array
	#define INMET_pop(a) \
		(dg__dynarr_check_notempty((a), "Don't pop an empty array!"), \
		 (a).p[((a).md.cnt > 0) ? (--(a).md.cnt) : 0])

	// returns the last element of the array
	#define INMET_last(a) \
		(dg__dynarr_check_notempty((a), "Don't call da_last() on an empty array!"), \
		 (a).p[((a).md.cnt > 0) ? ((a).md.cnt-1) : 0])

#elif (INMET_INDEX_CHECK_LEVEL == 0) || (INMET_INDEX_CHECK_LEVEL == 2)

	// removes and returns the last element of the array
	#define INMET_pop(a) \
		(dg__dynarr_check_notempty((a), "Don't pop an empty array!"), \
		 (a).p[--(a).md.cnt])

	// returns the last element of the array
	#define INMET_last(a) \
		(dg__dynarr_check_notempty((a), "Don't call da_last() on an empty array!"), \
		 (a).p[(a).md.cnt-1])

#else // invalid INMET_INDEX_CHECK_LEVEL
	#error Invalid index check level INMET_INDEX_CHECK_LEVEL (must be 0-3) !
#endif // INMET_INDEX_CHECK_LEVEL



#if (INMET_INDEX_CHECK_LEVEL == 2) || (INMET_INDEX_CHECK_LEVEL == 3)

	#define dg__dynarr_checkidx(a,i) \
		INMET_ASSERT((size_t)i < a.md.cnt, "index out of bounds!")

	// special case for insert operations: == cnt is also ok, insert will append then
	#define dg__dynarr_checkidxle(a,i) \
		INMET_ASSERT((size_t)i <= a.md.cnt, "index out of bounds!")

	#define dg__dynarr_check_notempty(a, msg) \
		INMET_ASSERT(a.md.cnt > 0, msg)

#elif (INMET_INDEX_CHECK_LEVEL == 0) || (INMET_INDEX_CHECK_LEVEL == 1)

	// no assertions that check if index is valid
	#define dg__dynarr_checkidx(a,i) (void)0
	#define dg__dynarr_checkidxle(a,i) (void)0

	#define dg__dynarr_check_notempty(a, msg) (void)0

#else // invalid INMET_INDEX_CHECK_LEVEL
	#error Invalid index check level INMET_INDEX_CHECK_LEVEL (must be 0-3) !
#endif // INMET_INDEX_CHECK_LEVEL


#if (INMET_INDEX_CHECK_LEVEL == 1) || (INMET_INDEX_CHECK_LEVEL == 3)

	// the given index, if valid, else 0
	#define dg__dynarr_idx(md,i) \
		(((size_t)(i) < md.cnt) ? (size_t)(i) : 0)

#elif (INMET_INDEX_CHECK_LEVEL == 0) || (INMET_INDEX_CHECK_LEVEL == 2)

	// don't check and default to 0 if invalid, but just use the given value
	#define dg__dynarr_idx(md,i) (size_t)(i)

#else // invalid INMET_INDEX_CHECK_LEVEL
	#error Invalid index check level INMET_INDEX_CHECK_LEVEL (must be 0-3) !
#endif // INMET_INDEX_CHECK_LEVEL

// ############## Implementation of non-inline functions ##############



#if 0
    // insert a single value v at index idx
    const bool insert(size_t const idx, T const& v) {
        return checkidxle(md, idx), insert(__unpack, idx, 1, false),
               data[idx(md, idx)] = v;
    }

    // insert n elements into a at idx, initialize them from array vals
    // doesn't return anything
    // ! vals (and all other args) is evaluated multiple times ! 
    void insertn(size_t const idx, T const* vals, size_t const n)
    {
        INMET_ASSERT(vals != NULL, "Don't pass NULL as vals to INMET_insertn!");
        checkidxle(md, idx);
        
        if (vals != NULL && insert(__unpack, idx, n, false)) {
            size_t i_= idx, v_ = 0, e_ = idx + n;
            
            while(i_ < e_)
                data[i_++] = vals[v_++];
        }
    }

    // insert n elements into a at idx and zeroe them with memset() 
    // returns pointer to first inserted element or NULL if out of memory
    T* insertn(size_t const idx, size_t const n, bool const zeroed = false) {
        return checkidxle(md, idx), insert(__unpack, idx, n, zeroed)
               ? &(data[idx(md, idx)]) : NULL;
    }

    // overwrite n elements of a, starting at idx, with values from array vals
    // doesn't return anything
    // ! vals (and all other args) is evaluated multiple times ! 
    void setn(size_t const idx, T const* vals, size_t const n) {
        INMET_ASSERT(vals != NULL, "Don't pass NULL as vals to INMET_setn!");
        size_t idx_ = idx;
        size_t end_ = idx_ + n;
        
        checkidx(md, idx_);
        checkidx(md, end_ - 1);
        
        if (vals != NULL && idx_ < md.cnt && end_ <= md.cnt) {
            size_t v_ = 0;
            
            while(idx_ < end_)
                data[idx_++] = vals[v_++];
        }
    }

    // delete the element at idx
    // n == 1: moving all following elements (=> keeps order)
    // n > 1:  move the last element there (=> doesn't keep order)
	// TODO: check whether idx+n < count?
    bool const del(size_t const idx, size_t const n = 1) {
        return checkidx(md, idx), _delete(__unpack, idx, n);
    }

    // delete the element at idx
    // n == 1: moving all following elements (=> keeps order)
    // n > 1:  move the last element there (=> doesn't keep order)
	// TODO: check whether idx+n < count?
    bool const delfast(size_t const idx, size_t const n = 1) {
        return checkidx(md, idx), _deletefast(__unpack, idx, n);
    }
#endif

#endif
