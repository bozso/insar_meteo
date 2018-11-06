#ifndef PYVECTOR_HH
#define PYVECTOR_HH

#include <stdlib.h> // size_t, malloc(), free(), realloc()
#include <string.h> // memset(), memcpy(), memmove()

#include "common_macros.hh"

#define __unpack (void**) &data, md

struct vector_meta {
    size_t cnt, cap, itemsize;
};


template<class T>
struct vector {
    T* data;
    vector_meta md;

    vector(): data(NULL), md.cnt(0), md.cap(0) md.itemsize(sizeof(T)) {};
    vector(T* buf, size_t const buf_cap);

    bool const init(size_t const buf_cap);
    void init(T* buf, size_t const buf_cap);
    
    bool const push(T const& elem) {
        return maybegrowadd(__unpack, 1) ? ((data[md.cnt++] = elem), false) : false;
    }
    
    void addn(T* vals, size_t const n)
    {
        DG_DYNARR_ASSERT(vals != NULL, "Don't pass NULL vals vals to addn!");
        if (vals != NULL && add(__unpack, n, false)) {
            size_t i_ = md.cnt - n, v_ = 0;
        
        while(i_ < md.cnt)
            data[ii_++] = vals[v_++];
	}

    // add n elements to the end of the array and zeroe them with memset()
    // returns pointer to first added element, NULL if out of memory (array is empty then)
    T* zeroed(size_t const n) {
        return add(__unpack, n, true) ? &(data[md.cnt - n]) : NULL;
    }

    // add n elements to the end of the array, which are uninitialized
    // returns pointer to first added element, NULL if out of memory (array is empty then)
    T* uninit(size_t const n) {
        return add(__unpack, n, false) ? &(data[md.cnt - n]) : NULL;
    }

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
        DG_DYNARR_ASSERT(vals != NULL, "Don't pass NULL as vals to dg_dynarr_insertn!");
        checkidxle(md, idx);
        
        if (vals != NULL && insert(__unpack, idx, n, false)) {
            size_t i_= idx, v_ = 0, e_ = idx + n;
            
            while(i_ < e_)
                data[i_++] = vals[v_++];
        }
    }

    // insert n elements into a at idx and zeroe them with memset() 
    // returns pointer to first inserted element or NULL if out of memory
    T* insertn_zeroed(size_t const idx, size_t const n) {
        return checkidxle(md, idx), insert(__unpack, idx, n, true)
               ? &(data[idx(md, idx)]) : NULL;
    }

    // insert n uninitialized elements into a at idx;
    // returns pointer to first inserted element or NULL if out of memory
    T* insertn_uninit(size_t const idx, size_t const n) {
        return checkidxle(md, idx), insert(__unpack, idx, n, false)
               ? &(data[idx(md, idx)]) : NULL;
    }

    // overwrite n elements of a, starting at idx, with values from array vals
    // doesn't return anything
    // ! vals (and all other args) is evaluated multiple times ! 
    void setn(size_t const idx, T const* vals, size_t const n) {
        DG_DYNARR_ASSERT(vals != NULL, "Don't pass NULL as vals to dg_dynarr_setn!");
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

    // removes all elements from the array, but does not free the buffer
    // (if you want to free the buffer too, just use free())
    void clear(a) {
        md.cnt = 0
    }

    // sets the logical number of elements in the array
    // if cnt > dg_dynarr_count(a), the logical count will be increased accordingly
    // and the new elements will be uninitialized
    bool const setcount(size_t const n) {
        maybegrow(__unpack, n) ? (md.cnt = n) : false;
    }

    // make sure the array can store cap elements without reallocating
    // logical count remains unchanged
    bool const reserve(size_t const cap) {
        return maybegrow(__unpack, cap);
    }

    // this makes sure a only uses as much memory as for its elements
    // => maybe useful if a used to contain a huge amount of elements,
    //    but you deleted most of them and want to free some memory
    // Note however that this implies an allocation and copying the remaining
    // elements, so only do this if it frees enough memory to be worthwhile!
#define dg_dynarr_shrink_to_fit(a) \
	dg__dynarr_shrink_to_fit(dg__dynarr_unp(a))
    
        
    ~vector() {
        free((void**) &data, md);
    }

    T& operator[](size_t ii) {
        return data[idx(ii)];
    }

    T const operator[](size_t ii) const {
        return data[idx(ii)];
    }
    
};


#if (DG_DYNARR_INDEX_CHECK_LEVEL == 2) || (DG_DYNARR_INDEX_CHECK_LEVEL == 3)
    
    template<class T>
    static void vector<T>::checkidx(size_t ii)
    {
        assert(ii < cnt, "index out of bounds");
    }

    template<class T>
    static void vector<T>::checkidxle(size_t ii)
    {
        assert(ii <= cnt, "index out of bounds");
    }

    template<class T>
    static void vector<T>::check_notempty(const char * msg)
    {
        assert(cnt > 0, msg);
    }

#elif (DG_DYNARR_INDEX_CHECK_LEVEL == 0) || (DG_DYNARR_INDEX_CHECK_LEVEL == 1)

    template<class T>
    static void* vector<T>::checkidx(size_t ii)
    {
        return (void) 0;
    }

    template<class T>
    static void* vector<T>::checkidxle(size_t ii)
    {
        return (void) 0;
    }

    template<class T>
    static void* vector<T>::check_notempty(const char * msg)
    {
        return (void) 0;
    }

#else // invalid DG_DYNARR_INDEX_CHECK_LEVEL
	#error Invalid index check level DG_DYNARR_INDEX_CHECK_LEVEL (must be 0-3) !
#endif // DG_DYNARR_INDEX_CHECK_LEVEL


#if (DG_DYNARR_INDEX_CHECK_LEVEL == 1) || (DG_DYNARR_INDEX_CHECK_LEVEL == 3)
    
    template<class T>
    size_t vector<T>::idx(size_t ii)
    {
        return (ii < cnt) ? ii : 0;
    }
    

#elif (DG_DYNARR_INDEX_CHECK_LEVEL == 0) || (DG_DYNARR_INDEX_CHECK_LEVEL == 2)
    #define idx(ii) (size_t) ii

#else // invalid DG_DYNARR_INDEX_CHECK_LEVEL
	#error Invalid index check level DG_DYNARR_INDEX_CHECK_LEVEL (must be 0-3) !
#endif // DG_DYNARR_INDEX_CHECK_LEVEL



#if (DG_DYNARR_INDEX_CHECK_LEVEL == 1) || (DG_DYNARR_INDEX_CHECK_LEVEL == 3)

	// removes and returns the last element of the array
	#define dg_dynarr_pop(a) \
		(dg__dynarr_check_notempty((a), "Don't pop an empty array!"), \
		 (a).p[((a).md.cnt > 0) ? (--(a).md.cnt) : 0])

	// returns the last element of the array
	#define dg_dynarr_last(a) \
		(dg__dynarr_check_notempty((a), "Don't call da_last() on an empty array!"), \
		 (a).p[((a).md.cnt > 0) ? ((a).md.cnt-1) : 0])

#elif (DG_DYNARR_INDEX_CHECK_LEVEL == 0) || (DG_DYNARR_INDEX_CHECK_LEVEL == 2)

	// removes and returns the last element of the array
	#define dg_dynarr_pop(a) \
		(dg__dynarr_check_notempty((a), "Don't pop an empty array!"), \
		 (a).p[--(a).md.cnt])

	// returns the last element of the array
	#define dg_dynarr_last(a) \
		(dg__dynarr_check_notempty((a), "Don't call da_last() on an empty array!"), \
		 (a).p[(a).md.cnt-1])

#else // invalid DG_DYNARR_INDEX_CHECK_LEVEL
	#error Invalid index check level DG_DYNARR_INDEX_CHECK_LEVEL (must be 0-3) !
#endif // DG_DYNARR_INDEX_CHECK_LEVEL

// returns the pointer *to* the last element of the array
// (in contrast to dg_dynarr_end() which returns a pointer *after* the last element)
// returns NULL if array is empty
#define dg_dynarr_lastptr(a) \
	(((a).md.cnt > 0) ? ((a).p + (a).md.cnt - 1) : NULL)


// get pointer to element at index idx (like &a.p[idx]), but with checks
// (unless you disabled them with #define DG_DYNARR_INDEX_CHECK_LEVEL 0)
// if index-checks are disabled, it returns NULL on invalid index (else it asserts() before returning)
#define dg_dynarr_getptr(a, idx) \
	(dg__dynarr_checkidx((a),(idx)), \
	 ((size_t)(idx) < (a).md.cnt) ? ((a).p+(size_t)(idx)) : NULL)

// returns a pointer to the first element of the array
// (together with dg_dynarr_end() you can do C++-style iterating)
#define dg_dynarr_begin(a) \
	((a).p)

// returns a pointer to the past-the-end element of the array
// Allows C++-style iterating, in case you're into that kind of thing:
// for(T *it=dg_dynarr_begin(a), *end=dg_dynarr_end(a); it!=end; ++it) foo(*it);
// (see dg_dynarr_lastptr() to get a pointer *to* the last element)
#define dg_dynarr_end(a) \
	((a).p + (a).md.cnt)


// returns (logical) number of elements currently in the array
#define dg_dynarr_count(a) \
	((a).md.cnt)

// get the current reserved capacity of the array
#define dg_dynarr_capacity(a) \
	((a).md.cap & DG__DYNARR_SIZE_T_ALL_BUT_MSB)

// returns 1 if the array is empty, else 0
#define dg_dynarr_empty(a) \
	((a).md.cnt == 0)

// returns 1 if the last (re)allocation when inserting failed (Out Of Memory)
//   or if the array has never allocated any memory yet, else 0
// deleting the contents when growing fails instead of keeping old may seem
// a bit uncool, but it's simple and OOM should rarely happen on modern systems
// anyway - after all you need to deplete both RAM and swap/pagefile.sys
// or deplete the address space, which /might/ happen with 32bit applications
// but probably not with 64bit (at least in the foreseeable future)
#define dg_dynarr_oom(a) \
	((a).md.cap == 0)


#if (DG_DYNARR_INDEX_CHECK_LEVEL == 2) || (DG_DYNARR_INDEX_CHECK_LEVEL == 3)

	#define dg__dynarr_checkidx(a,i) \
		DG_DYNARR_ASSERT((size_t)i < a.md.cnt, "index out of bounds!")

	// special case for insert operations: == cnt is also ok, insert will append then
	#define dg__dynarr_checkidxle(a,i) \
		DG_DYNARR_ASSERT((size_t)i <= a.md.cnt, "index out of bounds!")

	#define dg__dynarr_check_notempty(a, msg) \
		DG_DYNARR_ASSERT(a.md.cnt > 0, msg)

#elif (DG_DYNARR_INDEX_CHECK_LEVEL == 0) || (DG_DYNARR_INDEX_CHECK_LEVEL == 1)

	// no assertions that check if index is valid
	#define dg__dynarr_checkidx(a,i) (void)0
	#define dg__dynarr_checkidxle(a,i) (void)0

	#define dg__dynarr_check_notempty(a, msg) (void)0

#else // invalid DG_DYNARR_INDEX_CHECK_LEVEL
	#error Invalid index check level DG_DYNARR_INDEX_CHECK_LEVEL (must be 0-3) !
#endif // DG_DYNARR_INDEX_CHECK_LEVEL


#if (DG_DYNARR_INDEX_CHECK_LEVEL == 1) || (DG_DYNARR_INDEX_CHECK_LEVEL == 3)

	// the given index, if valid, else 0
	#define dg__dynarr_idx(md,i) \
		(((size_t)(i) < md.cnt) ? (size_t)(i) : 0)

#elif (DG_DYNARR_INDEX_CHECK_LEVEL == 0) || (DG_DYNARR_INDEX_CHECK_LEVEL == 2)

	// don't check and default to 0 if invalid, but just use the given value
	#define dg__dynarr_idx(md,i) (size_t)(i)

#else // invalid DG_DYNARR_INDEX_CHECK_LEVEL
	#error Invalid index check level DG_DYNARR_INDEX_CHECK_LEVEL (must be 0-3) !
#endif // DG_DYNARR_INDEX_CHECK_LEVEL

// ############## Implementation of non-inline functions ##############

#endif
