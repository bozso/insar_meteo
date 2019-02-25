if VECTOR_SUPPORT
TMP_TESTV = testmasimd testmasimd2 testsimdstatistics testsimdstatistics2 testuserfuncs
TMP_CHECKV = testmasimd testmasimd2 testsimdstatistics testsimdstatistics2 testuserfuncs
else
TMP_TESTV =
TMP_CHECKV =
endif

if BLAS_SUPPORT
TMP_TESTBLAS = testmablas
else
TMP_TESTBLAS =
endif

if LAPACK_SUPPORT
TMP_TESTLAPACK = testmalapack
else
TMP_TESTLAPACK =
endif

if COMPLEX_SUPPORT
TMP_TESTCOMPLEX = testcomplex
else
TMP_TESTCOMPLEX = 
endif

TESTS = testmarray				\
	testmmarray 				\
	teststatistics				\
	testpartialreduc     			\
	testmerge				\
	testmarrayiter              \
	testfvec testfvec2			\
	testfmatrix testfmatrix2		\
	testmatvec testmatvec2			\
	testmatmat testmatmat2			\
	testmarrayfvec                  \
	testgaussj testgaussj2			\
	testludecomp                    	\
	testsvdecomp                    	\
	testgaussmarquardt testgaussmarquardt2	\
	testpolynomfit testpolynomfit2		\
	testlmfit				\
	testieee				\
	testfitsio testfitsextensionio		\
	testexceptions				\
	testascio				\
	testmarrayio				\
	testutildate				\
	testutilregion				\
	testconfreader				\
	testrange				\
	testconvolve                \
	$(TMP_TESTV) $(TMP_TESTBLAS) $(TMP_TESTLAPACK) $(TMP_TESTCOMPLEX)


check_PROGRAMS = testmarray			\
	testmmarray 				\
	teststatistics				\
	testpartialreduc     			\
	testmerge                   		\
	testmarrayiter              \
	testfvec testfvec2			\
	testfmatrix testfmatrix2		\
	testmatvec testmatvec2			\
	testmatmat testmatmat2			\
	testmarrayfvec                  \
	testgaussj testgaussj2			\
	testludecomp                    	\
	testsvdecomp                    	\
	testgaussmarquardt testgaussmarquardt2	\
	testpolynomfit testpolynomfit2		\
	testlmfit				\
	testieee				\
	testfitsio testfitsextensionio		\
	testexceptions				\
	testascio				\
	testmarrayio				\
	testutildate				\
	testutilregion				\
	testconfreader				\
	testrange                   		\
	testconvolve                \
	$(TMP_TESTV) $(TMP_TESTBLAS) $(TMP_TESTLAPACK) $(TMP_TESTCOMPLEX)


testmarray_SOURCES    = testmarray.cpp
testmmarray_SOURCES   = testmmarray.cpp
teststatistics_SOURCES= teststatistics.cpp
testpartialreduc_SOURCES= testpartialreduc.cpp
testmerge_SOURCES     = testmerge.cpp
testmarrayiter_SOURCES= testmarrayiter.cpp
testfvec_SOURCES      = testfvec.cpp
testfvec2_SOURCES     = testfvec2.cpp
testfmatrix_SOURCES   = testfmatrix.cpp
testfmatrix2_SOURCES  = testfmatrix2.cpp
testmatvec_SOURCES    = testmatvec.cpp
testmatvec2_SOURCES   = testmatvec2.cpp
testmatmat_SOURCES    = testmatmat.cpp
testmatmat2_SOURCES   = testmatmat2.cpp
testmarrayfvec_SOURCES= testmarrayfvec.cpp
testgaussj_SOURCES    = testgaussj.cpp
testgaussj2_SOURCES   = testgaussj2.cpp
testgaussmarquardt_SOURCES  = testgaussmarquardt.cpp
testgaussmarquardt2_SOURCES = testgaussmarquardt2.cpp
testludecomp_SOURCES  = testludecomp.cpp
testsvdecomp_SOURCES  = testsvdecomp.cpp
testlmfit_SOURCES     = testlmfit.cpp
testpolynomfit_SOURCES  = testpolynomfit.cpp
testpolynomfit2_SOURCES = testpolynomfit2.cpp
testieee_SOURCES       = testieee.cpp
testfitsio_SOURCES     = testfitsio.cpp
testfitsextensionio_SOURCES    = testfitsextensionio.cpp
testexceptions_SOURCES = testexceptions.cpp
testascio_SOURCES      = testascio.cpp
testmarrayio_SOURCES   = testmarrayio.cpp
testutildate_SOURCES   = testutildate.cpp
testutilregion_SOURCES   = testutilregion.cpp
testconfreader_SOURCES = testconfreader.cpp
testrange_SOURCES      = testrange.cpp
testconvolve_SOURCES   = testconvolve.cpp

if VECTOR_SUPPORT
testmasimd_SOURCES = testmasimd.cpp
testmasimd2_SOURCES = testmasimd2.cpp
testsimdstatistics_SOURCES = testsimdstatistics.cpp
testsimdstatistics2_SOURCES = testsimdstatistics2.cpp
testuserfuncs_SOURCES = testuserfuncs.cpp
testmasimd_CXXFLAGS = $(CXXVECFLAGS)
testmasimd2_CXXFLAGS = $(CXXVECFLAGS)
testsimdstatistics_CXXFLAGS = $(CXXVECFLAGS)
testsimdstatistics2_CXXFLAGS = $(CXXVECFLAGS)
testuserfuncs_CXXFLAGS = $(CXXVECFLAGS)
endif


if APPLE_VECLIB
testmasimd_LDFLAGS = -framework Accelerate $(LDFLAGS)
testsimdstatistics_LDFLAGS = -framework Accelerate $(LDFLAGS)
endif

#testmaperf_SOURCES = testmaperf.cpp


if BLAS_SUPPORT
testmablas_SOURCES = testmablas.cpp
testmablas_LDFLAGS = $(LIBS)
endif

if LAPACK_SUPPORT
testmalapack_SOURCES = testmalapack.cpp
testmalapack_LDFLAGS = $(LIBS)
endif

if COMPLEX_SUPPORT
testcomplex_SOURCES = testcomplex.cpp
testcomplex_LDFLAGS = $(LIBS)
endif

testfitsio_LDADD = -L../ltl -lltl
testfitsextensionio_LDADD = -L../ltl -lltl
testascio_LDADD  = -L../ltl -lltl
testutildate_LDADD  = -L../ltl -lltl
testutilregion_LDADD  = -L../ltl -lltl
testconfreader_LDADD  = -L../ltl -lltl


AM_CPPFLAGS = -I../
CLEANFILES = test-64.fits test-32.fits test32.fits test16.fits test8.fits \
	     testextensionio.fits test_inmem_float.fits \
             testc.fits test-reg.fits test.map test.dat ltl-test.conf \
             testmarrayio0.dat testmarrayio1.dat testmarrayio2.dat

