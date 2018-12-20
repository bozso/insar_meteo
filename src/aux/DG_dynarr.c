// ######### Implementation-Details that are not part of the API ##########

#include <stdlib.h> // size_t, malloc(), free(), realloc()
#include <string.h> // memset(), memcpy(), memmove()

#include "DG_dynarr.h"
#include "common.h"

extern_begin

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


// the following functions are implemented inline, because they're quite short
// and mosty implemented in functions so the macros don't get too ugly

DG_DYNARR_INLINE void
dg__dynarr_init(void** p, dg__dynarr_md* md, void* buf, size_t buf_cap)
{
	*p = buf;
	md->cnt = 0;
	if(buf == NULL)  md->cap = 0;
	else md->cap = (DG__DYNARR_SIZE_T_MSB | buf_cap);
}

DG_DYNARR_INLINE int
dg__dynarr_maybegrow(void** arr, dg__dynarr_md* md, size_t itemsize, size_t min_needed)
{
	if((md->cap & DG__DYNARR_SIZE_T_ALL_BUT_MSB) >= min_needed)  return 1;
	else return dg__dynarr_grow(arr, md, itemsize, min_needed);
}

DG_DYNARR_INLINE int
dg__dynarr_maybegrowadd(void** arr, dg__dynarr_md* md, size_t itemsize, size_t num_add)
{
	size_t min_needed = md->cnt+num_add;
	if((md->cap & DG__DYNARR_SIZE_T_ALL_BUT_MSB) >= min_needed)  return 1;
	else return dg__dynarr_grow(arr, md, itemsize, min_needed);
}

DG_DYNARR_INLINE int
dg__dynarr_insert(void** arr, dg__dynarr_md* md, size_t itemsize, size_t idx, size_t n, int init0)
{
	// allow idx == md->cnt to append
	size_t oldCount = md->cnt;
	size_t newCount = oldCount+n;
	if(idx <= oldCount && dg__dynarr_maybegrow(arr, md, itemsize, newCount))
	{
		unsigned char* p = (unsigned char*)*arr; // *arr might have changed in dg__dynarr_grow()!
		// move all existing items after a[idx] to a[idx+n]
		if(idx < oldCount)  memmove(p+(idx+n)*itemsize, p+idx*itemsize, itemsize*(oldCount - idx));

		// if the memory is supposed to be zeroed, do that
		if(init0)  memset(p+idx*itemsize, 0, n*itemsize);

		md->cnt = newCount;
		return 1;
	}
	return 0;
}

DG_DYNARR_INLINE int
dg__dynarr_add(void** arr, dg__dynarr_md* md, size_t itemsize, size_t n, int init0)
{
	size_t cnt = md->cnt;
	if(dg__dynarr_maybegrow(arr, md, itemsize, cnt+n))
	{
		unsigned char* p = (unsigned char*)*arr; // *arr might have changed in dg__dynarr_grow()!
		// if the memory is supposed to be zeroed, do that
		if(init0)  memset(p+cnt*itemsize, 0, n*itemsize);

		md->cnt += n;
		return 1;
	}
	return 0;
}

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

#ifdef __cplusplus
} // extern "C"
#endif



// ############## Implementation of non-inline functions ##############

#ifdef DG_DYNARR_IMPLEMENTATION

// by default, C's malloc(), realloc() and free() is used to allocate/free heap memory.
// you can #define DG_DYNARR_MALLOC, DG_DYNARR_REALLOC and DG_DYNARR_FREE
// to provide alternative implementations like Win32 Heap(Re)Alloc/HeapFree
// 
#ifndef DG_DYNARR_MALLOC
	#define DG_DYNARR_MALLOC(elemSize, numElems)  malloc(elemSize*numElems)

	// oldNumElems is not used here, but maybe you need it for your allocator
	// to copy the old elements over
	#define DG_DYNARR_REALLOC(ptr, elemSize, oldNumElems, newCapacity) \
		realloc(ptr, elemSize*newCapacity);

	#define DG_DYNARR_FREE(ptr)  free(ptr)
#endif

// you can #define DG_DYNARR_OUT_OF_MEMORY to some code that will be executed
// if allocating memory fails
#ifndef DG_DYNARR_OUT_OF_MEMORY
	#define DG_DYNARR_OUT_OF_MEMORY  DG_DYNARR_ASSERT(0, "Out of Memory!");
#endif


#ifdef __cplusplus
extern "C" {
#endif

DG_DYNARR_DEF void
dg__dynarr_free(void** p, dg__dynarr_md* md)
{
	// only free memory if it doesn't point to external memory
	if(!(md->cap & DG__DYNARR_SIZE_T_MSB))
	{
		DG_DYNARR_FREE(*p);
		*p = NULL;
		md->cap = 0;
	}
	md->cnt = 0;
}


DG_DYNARR_DEF int
dg__dynarr_grow(void** arr, dg__dynarr_md* md, size_t itemsize, size_t min_needed)
{
	size_t cap = md->cap & DG__DYNARR_SIZE_T_ALL_BUT_MSB;

	DG_DYNARR_ASSERT(min_needed > cap, "dg__dynarr_grow() should only be called if storage actually needs to grow!");

	if(min_needed < DG__DYNARR_SIZE_T_MSB)
	{
		size_t newcap = (cap > 4) ? (2*cap) : 8; // allocate for at least 8 elements
		// make sure not to set DG__DYNARR_SIZE_T_MSB (unlikely anyway)
		if(newcap >= DG__DYNARR_SIZE_T_MSB)  newcap = DG__DYNARR_SIZE_T_MSB-1;
		if(min_needed > newcap)  newcap = min_needed;

		// the memory was allocated externally, don't free it, just copy contents
		if(md->cap & DG__DYNARR_SIZE_T_MSB)
		{
			void* p = DG_DYNARR_MALLOC(itemsize, newcap);
			if(p != NULL)  memcpy(p, *arr, itemsize*md->cnt);
			*arr = p;
		}
		else
		{
			void* p = DG_DYNARR_REALLOC(*arr, itemsize, md->cnt, newcap);
			if(p == NULL)  DG_DYNARR_FREE(*arr); // realloc failed, at least don't leak memory
			*arr = p;
		}

		// TODO: handle OOM by setting highest bit of count and keeping old data?

		if(*arr)  md->cap = newcap;
		else
		{
			md->cap = 0;
			md->cnt = 0;
			
			DG_DYNARR_OUT_OF_MEMORY ;
			
			return 0;
		}
		return 1;
	}
	DG_DYNARR_ASSERT(min_needed < DG__DYNARR_SIZE_T_MSB, "Arrays must stay below SIZE_T_MAX/2 elements!");
	return 0;
}

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

extern_end

#endif // DG_DYNARR_IMPLEMENTATION
