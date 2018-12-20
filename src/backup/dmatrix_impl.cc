#include "dmatrix.hh"

// you can #define INMET_OUT_OF_MEMORY to some code that will be executed
// if allocating memory fails
#ifndef INMET_OUT_OF_MEMORY
	#define INMET_OUT_OF_MEMORY  INMET_ASSERT(0, "Out of Memory!");
#endif

using namespace dmat_impl;

namespace dmat_impl {

INMET_DEF static bool
grow(void** arr, dmat_meta& md, size_t const min_needed);

INMET_INLINE void
init(void** p, dmat_meta& meta, void const* buf, size_t const buf_cap)
{
	*p = buf;
	md.cnt = 0;
	
    if (buf == NULL)
        md.cap = 0;
	else
        md.cap = (dmat_impl::size_t_msb | buf_cap);
}


INMET_INLINE bool
maybegrow(void** arr, dmat_meta& md, size_t const min_needed)
{
	if ((md.cap & dmat_impl::size_t_all_but_msb) >= min_needed)
        return true;
	else
        return grow(arr, md, min_needed);
}


INMET_INLINE bool
maybegrowadd(void** arr, dmat_meta& md, size_t const num_add)
{
	size_t min_needed = md.cnt + num_add;
    
	if ((md.cap & dmat_impl::size_t_all_but_msb) >= min_needed)
        return true;
	else
        return grow(arr, md, min_needed);
}


INMET_DEF void
free(void** p, dmat_meta& md)
{
	// only free memory if it doesn't point to external memory
	if (not (md.cap & dmat_impl::size_t_msb)) {
		Mem_Del(*p);
		*p = NULL;
		md.cap = 0;
	}
	md.cnt = 0;
}


INMET_DEF void
shrink_to_fit(void** arr, dmat_meta& md)
{
	// only do this if we allocated the memory ourselves
	if (not(md.cap & dmat_impl::size_t_msb)) {
		size_t cnt = md.cnt;
		if (cnt == 0) {
            free(arr, md);
        }
		else if((md->cap & dmat_impl::size_t_all_but_msb) > cnt) {
			void* p = Mem_New(unsigned char, cnt * md.itemsize);
	
    		if(p != NULL) {
				memcpy(p, *arr, cnt*itemsize);
				md->cap = cnt;
				Mem_Del(*arr);
				*arr = p;
			}
		}
	}
}


INMET_INLINE bool
insert(void** arr, dmat_meta& md, size_t idx, size_t n, bool const init0)
{
	// allow idx == md->cnt to append
	size_t oldCount = md.cnt;
	size_t newCount = oldCount + n;
    
	if (idx <= oldCount && maybegrow(arr, md, newCount)) {
		// *arr might have changed in dg__dynarr_grow()!
        unsigned char* p = (unsigned char*)*arr;
		
        // move all existing items after a[idx] to a[idx+n]
		if(idx < oldCount)
            memmove(p + (idx + n) * md.itemsize, p + idx * md.itemsize,
                    md.itemsize * (oldCount - idx));

		// if the memory is supposed to be zeroed, do that
		if (init0)
            memset(p + idx * md.itemsize, 0, n * md.itemsize);

		md.cnt = newCount;
		return true;
	}
    
	return false;
}


INMET_INLINE bool
add(void** arr, dmat_meta& md, size_t n, bool const init0)
{
	size_t cnt = md.cnt;
	
    if (maybegrow(arr, md, cnt + n)) {
        // *arr might have changed in dg__dynarr_grow()!
		unsigned char* p = (unsigned char*)*arr; 
		
        // if the memory is supposed to be zeroed, do that
		if(init0)
            memset(p + cnt * md.itemsize, 0,  n * md.itemsize);

		md.cnt += n;
		return true;
	}
	return false;
}


INMET_INLINE bool
addn(void * vals, void** arr, dmat_meta& md, size_t const n, bool const init0)
{
    INMET_ASSERT(vals != NULL, "Don't pass NULL vals vals to addn!");
    
}


INMET_INLINE void
del(void** arr, dmat_meta& md, size_t const idx, size_t const n)
{
	size_t cnt = md.cnt;
	
    if(idx < cnt) {
		if (idx + n >= cnt) {
            // removing last element(s) => just reduce count
            md.cnt = idx; 
		} else {
			unsigned char* p = (unsigned char*)*arr;
			
            // move all items following a[idx+n] to a[idx]
			memmove(p + md.itemsize * idx, p + md.itemsize * (idx + n),
                    md.itemsize * (cnt - (idx + n)));
			
            md.cnt -= n;
		}
	}
}


INMET_INLINE void
delfast(void** arr, dmat_meta& md, size_t const idx, size_t const n)
{
	size_t cnt = md.cnt;
	
    if(idx < cnt) {
		if(idx + n >= cnt) {
            // removing last element(s) => just reduce count
            md.cnt = idx; 
		} else {
			unsigned char* p = (unsigned char*)*arr;
			
            // copy the last n items to a[idx] - but handle the case that
			// the array has less than n elements left after the deleted elements
			size_t numItemsAfterDeleted = cnt - (idx + n);
            size_t m = (n < numItemsAfterDeleted) ? n : numItemsAfterDeleted;
			
            memcpy(p + md.itemsize * idx, p + md.itemsize * (cnt - m),
                   md.itemsize * m);
			md.cnt -= n;
		}
	}
}


INMET_DEF static bool
grow(void** arr, dmat_meta& md, size_t const min_needed)
{
	size_t cap = md.cap & dmat_impl::size_t_all_but_msb;

	INMET_ASSERT(min_needed > cap, "grow() should only be called if "
                                   "storage actually needs to grow!");

	if(min_needed < dmat_impl::size_t_msb) {
        // allocate for at least 8 rows
        size_t _cap = cap * md.ncol;
		size_t newcap = (_cap > 4) ? (2 * _cap) : 8;
        
		// make sure not to set size_t_msb (unlikely anyway)
		if (newcap >= dmat_impl::size_t_msb)
            newcap = dmat_impl::size_t_msb - 1;
		
        if (min_needed > newcap)
            newcap = min_needed;

		// the memory was allocated externally, don't free it, just copy contents
		if (md.cap & dmat_impl::size_t_msb) {
			void* p = Mem_New(void, md.itemsize * newcap);
			
            if (p != NULL)
                memcpy(p, *arr, md.itemsize * md.cnt);
			*arr = p;
		}
        else {
			void* p = Mem_Resize(*arr, void, md.itemsize * newcap);
			
            // realloc failed, at least don't leak memory
            if (p == NULL)
                Mem_Free(*arr);
			
            *arr = p;
		}

		// TODO: handle OOM by setting highest bit of count and keeping old data?

		if (*arr)
            md.cap = newcap;
		else {
			md.cap = 0;
			md.cnt = 0;
			
			INMET_OUT_OF_MEMORY;
			
			return false;
		}
		return true;
	}
	INMET_ASSERT(min_needed < dmat_impl::size_t_msb, "Arrays must stay "
                 "below SIZE_T_MAX/2 elements!");
	return false;
}


#if (INMET_INDEX_CHECK_LEVEL == 2) || (INMET_INDEX_CHECK_LEVEL == 3)
    
static void checkidx(dmat_meta const& md, size_t const ii) {
    assert(ii < md.cnt, "index out of bounds");
}


static void checkidxle(dmat_meta const& md, size_t const ii) {
    assert(ii <= md.cnt, "index out of bounds");
}


static void check_notempty(dmat_meta const& md,  char const* msg) {
    assert(md.cnt > 0, msg);
}


#elif (INMET_INDEX_CHECK_LEVEL == 0) || (INMET_INDEX_CHECK_LEVEL == 1)


static void* checkidx(dmat_meta const& md, size_t const ii) {
    return (void) 0;
}


static void* checkidxle(dmat_meta const& md, size_t const ii) {
    return (void) 0;
}


static void* check_notempty(dmat_meta const& md,  char const* msg) {
    return (void) 0;
}


#else // invalid INMET_INDEX_CHECK_LEVEL
	#error Invalid index check level INMET_INDEX_CHECK_LEVEL (must be 0-3) !
#endif // INMET_INDEX_CHECK_LEVEL


#if (INMET_INDEX_CHECK_LEVEL == 1) || (INMET_INDEX_CHECK_LEVEL == 3)

    INMET_INLINE size_t
    idx(dmat_meta const& md, size_t const ii) { return (ii < md.cnt) ? ii : 0; }

#elif (INMET_INDEX_CHECK_LEVEL == 0) || (INMET_INDEX_CHECK_LEVEL == 2)
    INMET_INLINE size_t
    idx(dmat_meta const& md, size_t const ii) { return ii; }

#else // invalid INMET_INDEX_CHECK_LEVEL
	#error Invalid index check level INMET_INDEX_CHECK_LEVEL (must be 0-3) !
#endif // INMET_INDEX_CHECK_LEVEL

}
