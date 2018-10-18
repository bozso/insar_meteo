#ifndef PYVECTOR_HH
#define PYVECTOR_HH

#include <stddef.h>

#include "common_macros.hh"

template<class T>
struct vector {
    T* data;
    size_t cnt;
    size_t cap;
    
    vector(): data(NULL), cnt(0), cap(0) {};
    bool const init(size_t const buf_cap);
    void init(T* buf, size_t const buf_cap);
    
    bool const push(T const& elem);

    bool const add(T* vals, size_t const n);
    T* addn(size_t const n, bool const init0);
    
    ~vector();
};


#ifdef __INMET_IMPL

static const size_t DG__DYNARR_SIZE_T_MSB = ((size_t)1) << (sizeof(size_t)*8 - 1);
static const size_t DG__DYNARR_SIZE_T_ALL_BUT_MSB = (((size_t)1) << (sizeof(size_t)*8 - 1))-1;


template<class T>
bool vector<T>::init(size_t buf_cap)
{
    if ((data = Mem_New(T, buf_cap)) == NULL) {
        pyexc(PyExc_MemoryError, "Failed to allocate memory!");
        return true;
    }
    
    cnt = 0;
    cap = buf_cap;
    
    return false;
}


template<class T>
void vector<T>::init(vec, T* buf, size_t buf_cap)
{
    data = buf;
    cnt= 0;
    cap = (DG__DYNARR_SIZE_T_MSB | buf_cap);
}


template<class T>
vector<T>::~vector()
{
	// only free memory if it doesn't point to external memory
	if(not (cap & DG__DYNARR_SIZE_T_MSB)) {
		Mem_Del(data);
		data = NULL;
		cap = 0;
	}
	cnt = 0;
}


template<class T>
DG_DYNARR_DEF
static bool grow(vector<T> *vec, size_t min_needed)
{
	size_t _cap = vec->cap & DG__DYNARR_SIZE_T_ALL_BUT_MSB;

	DG_DYNARR_ASSERT(min_needed > _cap, "dg__dynarr_grow() should only be "
                                        "called if storage actually needs to grow!");

	if(min_needed < DG__DYNARR_SIZE_T_MSB) {
        // allocate for at least 8 elements
		size_t newcap = (_cap > 4) ? (2 * _cap) : 8; 
		
        // make sure not to set DG__DYNARR_SIZE_T_MSB (unlikely anyway)
        if (newcap >= DG__DYNARR_SIZE_T_MSB)
            newcap = DG__DYNARR_SIZE_T_MSB - 1;
		
        if (min_needed > newcap)
            newcap = min_needed;

		// the memory was allocated externally, don't free it, just copy contents
		if(vec->cap & DG__DYNARR_SIZE_T_MSB) {
			T* p = Mem_New(T, newcap);
			if (p != NULL)
                memcpy(p, vec->data, sizeof(T)*cnt);
			vec->data = p;
		}
		else {
			T* p = Mem_Resize(vec->data, T, newcap);
			
            // realloc failed, at least don't leak memory
            if (p == NULL)
                Mem_Del(vec->data);
			vec->data = p;
		}

		// TODO: handle OOM by setting highest bit of count and keeping old data?

		if(vec->data)
            vec->cap = newcap;
		else {
			vec->cap = 0;
			vec->cnt = 0;
			
			DG_DYNARR_OUT_OF_MEMORY ;
			
			return false;
		}
		return true;
	}
    
	DG_DYNARR_ASSERT(min_needed < DG__DYNARR_SIZE_T_MSB, "Arrays must stay "
                     "below SIZE_T_MAX / 2 elements!");
	return false;
}


template<class T>
bool vector<T>::push(const T elem)
{
    return maybegrowadd(this, 1) ? ((data[cnt++] = elem),0) : false
}


template<class T>
DG_DYNARR_INLINE
static bool maybegrowadd(vector<T> *vec, const size_t num_add)
{
	size_t min_needed = vec->cnt + num_add;
	if ((vec.cap & DG__DYNARR_SIZE_T_ALL_BUT_MSB) >= min_needed)
        return true;
	else
        return grow(vec, min_needed);
}


template<class T>
DG_DYNARR_INLINE
static bool maybegrow(vector<T> *vec, const size_t min_needed)
{
	if ((vec->cap & DG__DYNARR_SIZE_T_ALL_BUT_MSB) >= min_needed)
        return true;
	else
        return grow(vec, min_needed);
}


template<class T>
DG_DYNARR_INLINE bool vector<T>::add(const size_t n, const bool init0)
{
	if (maybegrow(this, cnt + n)) {
        // data might have changed in grow()!
		unsigned char* p = (unsigned char*) data; 
		
        // if the memory is supposed to be zeroed, do that
		if(init0)
            memset(p + cnt * sizeof(T), 0, n * sizeof(T));

		cnt += n;
		return true;
	}
	return false;
}


// append n elements to a and initialize them from array vals, doesn't return anything
// ! vals (and all other args) are evaluated multiple times !
template<class T>
bool vector<T>::add(T* vals, const size_t n)
{
    DG_DYNARR_ASSERT(vals != NULL, "Don't pass NULL vals to addn!");
    if (vals != NULL && add(n, 0)) {
        size_t i_= cnt - n, v_ = 0;
        
        while(i_ < cnt)
            data[i_++] = vals[v_++];
	}
}

// add n elements to the end of the array and zeroe them with memset()
// returns pointer to first added element, NULL if out of memory (array is empty then)
template<class T>
T* vector<T>::addn(size_t n, bool init0)
{
    return add(n, init0) ? data[cnt - n] : NULL;
}

template<class T>
DG_DYNARR_INLINE bool insert(vector<T> *vec, const size_t idx, const size_t n, const bool init0)
{
	// allow idx == md->cnt to append
	size_t oldCount = vec->cnt;
	size_t newCount = oldCount + n;
	if (idx <= oldCount && maybegrow(vec, newCount)) {
        // data might have changed in grow()!
		unsigned char *p = (unsigned char*) vec->data; 
		
        // move all existing items after a[idx] to a[idx+n]
		if(idx < oldCount)
            memmove(p + (idx + n) * sizeof(T), p + idx * sizeof(T),
                    sizeof(T) * (oldCount - idx));

		// if the memory is supposed to be zeroed, do that
		if (init0)
            memset(p + idx * sizeof(T), 0, n * sizeof(T));

		vec->cnt = newCount;
		return true;
	}
	return false;
}



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

template<class T>
bool vector<T>::insert(size_t ii, T& value)
{
    return checkidxle(idx), _insert(idx, 1, false),
           data[idx(ii)] = value;
}

template<class T>
bool vector<T>::insert(size_t ii, T* values, size_t n)
{
	DG_DYNARR_ASSERT(vals != NULL, "Don't pass NULL as vals to dg_dynarr_insertn!");
	checkidxle(a, ii);
	
    if (vals != NULL && _insert(idx, n, 0)) {
		size_t i_ = ii, v_ = 0, e_ = ii + n;
	
    	while(i_ < e_)
            data[i_++] = vals[v_++];
    }
}


template<class T>
T* vector<T>::insert(size_t ii, size_t n, bool init0)
{
	return checkidxle(ii), _insert(ii, n, init0)
           ? data[idx(ii)] : NULL
    
    
}


template<class T>
T& vector<T>::operator[](size_t ii)
{
    return data[idx(ii)];
}


template<class T>
void vector<T>::set(size_t ii, T* values, size_t n)
{
	DG_DYNARR_ASSERT(vals != NULL, "Don't pass NULL as vals to dg_dynarr_setn!");
	
    size_t idx_= ii, end_ = idx_ + n;
    checkidx(idx_);
    checkidx(end_ - 1);
    
	if (vals != NULL && idx_ < cnt && end_ <= cnt) {
		size_t v_ = 0;
		while (idx_ < end_)
            data[idx_++] = values[v_++];
    }
}


template<class T>
bool vector<T>::del(size_t ii, size_t n)
{
	// TODO: check whether idx+n < count?
    return checkidx(ii), _delete(ii, n);
}

template<class T>
bool vector<T>::delfast(size_t ii, size_t n)
{
	// TODO: check whether idx+n < count?
    return checkidx(ii), _delete_fast(ii, n);
}


template<class T>

// removes all elements from the array, but does not free the buffer
// (if you want to free the buffer too, just use dg_dynarr_free())
#define dg_dynarr_clear(a) \
	((a).md.cnt=0)

// sets the logical number of elements in the array
// if cnt > dg_dynarr_count(a), the logical count will be increased accordingly
// and the new elements will be uninitialized
#define dg_dynarr_setcount(a, n) \
	(dg__dynarr_maybegrow(dg__dynarr_unp(a), (n)) ? ((a).md.cnt = (n)) : 0)

// make sure the array can store cap elements without reallocating
// logical count remains unchanged
#define dg_dynarr_reserve(a, cap) \
	dg__dynarr_maybegrow(dg__dynarr_unp(a), (cap))

// this makes sure a only uses as much memory as for its elements
// => maybe useful if a used to contain a huge amount of elements,
//    but you deleted most of them and want to free some memory
// Note however that this implies an allocation and copying the remaining
// elements, so only do this if it frees enough memory to be worthwhile!
#define dg_dynarr_shrink_to_fit(a) \
	dg__dynarr_shrink_to_fit(dg__dynarr_unp(a))


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

// get element at index idx (like a.p[idx]), but with checks
// (unless you disabled them with #define DG_DYNARR_INDEX_CHECK_LEVEL 0)
#define dg_dynarr_get(a, idx) \
	(dg__dynarr_checkidx((a),(idx)), (a).p[dg__dynarr_idx((a).md, (idx))])

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


// sort a using the given qsort()-comparator cmp
// (just a slim wrapper around qsort())
#define dg_dynarr_sort(a, cmp) \
	qsort((a).p, (a).md.cnt, sizeof((a).p[0]), (cmp))


// ######### Implementation-Details that are not part of the API ##########

#include <stdlib.h> // size_t, malloc(), free(), realloc()
#include <string.h> // memset(), memcpy(), memmove()

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	size_t cnt; // logical number of elements
	size_t cap; // cap & DG__DYNARR_SIZE_T_ALL_BUT_MSB is actual capacity (in elements, *not* bytes!)
		// if(cap & DG__DYNARR_SIZE_T_MSB) the current memory is not allocated by dg_dynarr,
		// but was set with dg_dynarr_init_external()
		// that's handy to give an array a base-element storage on the stack, for example
		// TODO: alternatively, we could introduce a flag field to this struct and use that,
		//       so we don't have to calculate & everytime cap is needed
} dg__dynarr_md;

// I used to have the following in an enum, but MSVC assumes enums are always 32bit ints
static const size_t DG__DYNARR_SIZE_T_MSB = ((size_t)1) << (sizeof(size_t)*8 - 1);
static const size_t DG__DYNARR_SIZE_T_ALL_BUT_MSB = (((size_t)1) << (sizeof(size_t)*8 - 1))-1;

// "unpack" the elements of an array struct for use with helper functions
// (to void** arr, dg__dynarr_md* md, size_t itemsize)
#define dg__dynarr_unp(a) \
	(void**)&(a).p, &(a).md, sizeof((a).p[0])

// MSVC warns about "conditional expression is constant" when using the
// do { ... } while(0) idiom in macros.. 
#ifdef _MSC_VER
  #if _MSC_VER >= 1400 // MSVC 2005 and newer
    // people claim MSVC 2005 and newer support __pragma, even though it's only documented
    // for 2008+ (https://msdn.microsoft.com/en-us/library/d9x1s805%28v=vs.90%29.aspx)
    // the following workaround is based on
    // http://cnicholson.net/2009/03/stupid-c-tricks-dowhile0-and-c4127/
    #define DG__DYNARR_WHILE0 \
      __pragma(warning(push)) \
      __pragma(warning(disable:4127)) \
      while(0) \
      __pragma(warning(pop))
  #else // older MSVC versions don't support __pragma - I heard this helps for them
    #define DG__DYNARR_WHILE0  while(0,0)
  #endif

#else // other compilers

	#define DG__DYNARR_WHILE0  while(0)

#endif // _MSC_VER


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

// the functions allocating/freeing memory are not implemented inline, but
// in the #ifdef DG_DYNARR_IMPLEMENTATION section
// one reason is that dg__dynarr_grow has the most code in it, the other is
// that windows has weird per-dll heaps so free() or realloc() should be
// called from code in the same dll that allocated the memory - these kind
// of wrapper functions that end up compiled into the exe or *one* dll
// (instead of inline functions compiled into everything) should ensure that.

DG_DYNARR_DEF void
dg__dynarr_free(void** p, dg__dynarr_md* md);

DG_DYNARR_DEF void
dg__dynarr_shrink_to_fit(void** arr, dg__dynarr_md* md, size_t itemsize);

// grow array to have enough space for at least min_needed elements
// if it fails (OOM), the array will be deleted, a.p will be NULL, a.md.cap and a.md.cnt will be 0
// and the functions returns 0; else (on success) it returns 1
DG_DYNARR_DEF int
dg__dynarr_grow(void** arr, dg__dynarr_md* md, size_t itemsize, size_t min_needed);







DG_DYNARR_INLINE void
dg__dynarr_delete(void** arr, dg__dynarr_md* md, size_t itemsize, size_t idx, size_t n)
{
	size_t cnt = md->cnt;
	if(idx < cnt)
	{
		if(idx+n >= cnt)  md->cnt = idx; // removing last element(s) => just reduce count
		else
		{
			unsigned char* p = (unsigned char*)*arr;
			// move all items following a[idx+n] to a[idx]
			memmove(p+itemsize*idx, p+itemsize*(idx+n), itemsize*(cnt - (idx+n)));
			md->cnt -= n;
		}
	}
}

DG_DYNARR_INLINE void
dg__dynarr_deletefast(void** arr, dg__dynarr_md* md, size_t itemsize, size_t idx, size_t n)
{
	size_t cnt = md->cnt;
	if(idx < cnt)
	{
		if(idx+n >= cnt)  md->cnt = idx; // removing last element(s) => just reduce count
		else
		{
			unsigned char* p = (unsigned char*)*arr;
			// copy the last n items to a[idx] - but handle the case that
			// the array has less than n elements left after the deleted elements
			size_t numItemsAfterDeleted = cnt - (idx+n);
			size_t m = (n < numItemsAfterDeleted) ? n : numItemsAfterDeleted;
			memcpy(p+itemsize*idx, p+itemsize*(cnt - m), itemsize*m);
			md->cnt -= n;
		}
	}
}

#endif // DG__DYNARR_H


// ############## Implementation of non-inline functions ##############

#ifdef DG_DYNARR_IMPLEMENTATION

// by default, C's malloc(), realloc() and free() is used to allocate/free heap memory.
// you can #define DG_DYNARR_MALLOC, DG_DYNARR_REALLOC and DG_DYNARR_FREE
// to provide alternative implementations like Win32 Heap(Re)Alloc/HeapFree
// 

// you can #define DG_DYNARR_OUT_OF_MEMORY to some code that will be executed
// if allocating memory fails
#ifndef DG_DYNARR_OUT_OF_MEMORY
	#define DG_DYNARR_OUT_OF_MEMORY  DG_DYNARR_ASSERT(0, "Out of Memory!");
#endif


DG_DYNARR_DEF void
dg__dynarr_shrink_to_fit(void** arr, dg__dynarr_md* md, size_t itemsize)
{
	// only do this if we allocated the memory ourselves
	if(!(md->cap & DG__DYNARR_SIZE_T_MSB))
	{
		size_t cnt = md->cnt;
		if(cnt == 0)  dg__dynarr_free(arr, md);
		else if((md->cap & DG__DYNARR_SIZE_T_ALL_BUT_MSB) > cnt)
		{
			void* p = DG_DYNARR_MALLOC(itemsize, cnt);
			if(p != NULL)
			{
				memcpy(p, *arr, cnt*itemsize);
				md->cap = cnt;
				DG_DYNARR_FREE(*arr);
				*arr = p;
			}
		}
	}
}

#endif

#endif
