#ifndef COMMON_MACROS_HH
#define COMMON_MACROS_HH

// some functions are inline, in case your compiler doesn't like "static inline"
// but wants "__inline__" or something instead, #define DG_DYNARR_INLINE accordingly.
#ifndef INMET_INLINE
	// for pre-C99 compilers you might have to use something compiler-specific (or maybe only "static")
	#ifdef _MSC_VER
		#define INMET_INLINE static __inline
	#else
		#define INMET_INLINE static inline
	#endif
#endif


// if you want to prepend something to the non inline (DG_DYNARR_INLINE) functions,
// like "__declspec(dllexport)" or whatever, #define DG_DYNARR_DEF
#ifndef INMET_DEF
	// by defaults it's empty.
	#define INMET_DEF
#endif


#ifndef DG_DYNARR_MALLOC
	#define Mem_New(elem_type, n_elem) (elem_type*) malloc(sizeof(elem_type) * n_elem)

	#define Mem_Resize(ptr, elem_type, new_n_elem) realloc(ptr, sizeof(elem_type) * new_n_elem)

	#define Mem_Del(ptr) free(ptr)
#endif

#endif
