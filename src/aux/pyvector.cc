#include "pyvector.hh"

// you can #define DG_DYNARR_OUT_OF_MEMORY to some code that will be executed
// if allocating memory fails
#ifndef DG_DYNARR_OUT_OF_MEMORY
	#define DG_DYNARR_OUT_OF_MEMORY  DG_DYNARR_ASSERT(0, "Out of Memory!");
#endif


static const size_t DG__DYNARR_SIZE_T_MSB = ((size_t)1) << (sizeof(size_t)*8 - 1);
static const size_t DG__DYNARR_SIZE_T_ALL_BUT_MSB = (((size_t)1) << (sizeof(size_t)*8 - 1))-1;

DG_DYNARR_DEF static bool
grow(void** arr, vector_meta& md, size_t min_needed);


DG_DYNARR_INLINE void
init(void** p, vector_meta& meta, void const* buf, size_t const buf_cap)
{
	*p = buf;
	md.cnt = 0;
	
    if (buf == NULL)
        md.cap = 0;
	else
        md.cap = (DG__DYNARR_SIZE_T_MSB | buf_cap);
}


// USED
DG_DYNARR_INLINE bool
maybegrow(void** arr, vector_meta& md, size_t const min_needed)
{
	if ((md.cap & DG__DYNARR_SIZE_T_ALL_BUT_MSB) >= min_needed)
        return true;
	else
        return grow(arr, md, min_needed);
}

// USED

DG_DYNARR_INLINE bool
maybegrowadd(void** arr, vector_meta& md, size_t const num_add)
{
	size_t min_needed = md.cnt + num_add;
    
	if ((md.cap & DG__DYNARR_SIZE_T_ALL_BUT_MSB) >= min_needed)
        return true;
	else
        return grow(arr, md, min_needed);
}

// USED
DG_DYNARR_DEF void
free(void** p, vector_meta& md)
{
	// only free memory if it doesn't point to external memory
	if (not(md.cap & DG__DYNARR_SIZE_T_MSB)) {
		PyMem_Free(*p);
		*p = NULL;
		md.cap = 0;
	}
	md.cnt = 0;
}


// USED
DG_DYNARR_DEF void
shrink_to_fit(void** arr, vector_meta& md)
{
	// only do this if we allocated the memory ourselves
	if (not(md.cap & DG__DYNARR_SIZE_T_MSB)) {
		size_t cnt = md.cnt;
        
		if (cnt == 0)
            free(arr, md);
		else if ((md.cap & DG__DYNARR_SIZE_T_ALL_BUT_MSB) > cnt) {
			size_t tmp = md.itemsize * md.cnt;
            
            void* p = PyMem_Malloc(tmp);
			
            if(p != NULL) {
				memcpy(p, *arr, tmp);
				md.cap = cnt;
				PyMem_Free(*arr);
				*arr = p;
			}
		}
	}
}

// USED
DG_DYNARR_INLINE bool
insert(void** arr, vector_meta& md, size_t idx, size_t n, bool init0)
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


// USED
DG_DYNARR_INLINE bool
add(void** arr, vector_meta& md, size_t n, bool init0)
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

// USED
DG_DYNARR_INLINE void
_delete(void** arr, vector_meta& md, size_t idx, size_t n)
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

// USED
DG_DYNARR_INLINE void
_deletefast(void** arr, vector_meta& md, size_t idx, size_t n)
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


DG_DYNARR_DEF static bool
grow(void** arr, vector_meta& md, size_t min_needed)
{
	size_t cap = md.cap & DG__DYNARR_SIZE_T_ALL_BUT_MSB;

	DG_DYNARR_ASSERT(min_needed > cap, "grow() should only be called if "
                                       "storage actually needs to grow!");

	if(min_needed < DG__DYNARR_SIZE_T_MSB) {
        // allocate for at least 8 elements
		size_t newcap = (cap > 4) ? (2 * cap) : 8;
        
		// make sure not to set DG__DYNARR_SIZE_T_MSB (unlikely anyway)
		if (newcap >= DG__DYNARR_SIZE_T_MSB)
            newcap = DG__DYNARR_SIZE_T_MSB - 1;
		
        if (min_needed > newcap)
            newcap = min_needed;

		// the memory was allocated externally, don't free it, just copy contents
		if (md.cap & DG__DYNARR_SIZE_T_MSB) {
			void* p = PyMem_Malloc(md.itemsize * newcap);
			
            if (p != NULL)
                memcpy(p, *arr, md.itemsize * md.cnt);
			*arr = p;
		} else {
			void* p = PyMem_Realloc(*arr, itemsize * newcap);
			
            // realloc failed, at least don't leak memory
            if (p == NULL)
                PyMem_Free(*arr); 
			
            *arr = p;
		}

		// TODO: handle OOM by setting highest bit of count and keeping old data?

		if (*arr)
            md.cap = newcap;
		else {
			md.cap = 0;
			md.cnt = 0;
			
			DG_DYNARR_OUT_OF_MEMORY ;
			
			return false;
		}
		return true;
	}
	DG_DYNARR_ASSERT(min_needed < DG__DYNARR_SIZE_T_MSB, "Arrays must stay "
                     "below SIZE_T_MAX/2 elements!");
	return false;
}


DG_DYNARR_DEF void
_shrink_to_fit(void** arr, vector_meta& md)
{
	// only do this if we allocated the memory ourselves
	if (not(md.cap & DG__DYNARR_SIZE_T_MSB)) {
		size_t cnt = md.cnt;
		if (cnt == 0)
            free(arr, md);
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
