#include <cstdlib>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <climits>
#include <cmath>
#include <ctime>

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <new>
#include <limits>
#include <algorithm>
#include <complex>
#include <vector>
#include <utility>
#include <map>


#if ( defined(__unix__) || defined(__unix) || defined(_POSIX_C_SOURCE) || (defined(__APPLE__) && defined(__MACH__)) ) && !defined(_WIN32)
  #include <unistd.h>
#endif


#if (defined(_POSIX_C_SOURCE) && (_POSIX_C_SOURCE >= 200112L))
  #include <sys/time.h>
#endif


#include "armadillo_bits/compiler_extra.hpp"
#include "armadillo_bits/config.hpp"
#include "armadillo_bits/compiler_setup.hpp"


#if defined(ARMA_USE_CXX11)
  #include <initializer_list>
  #include <cstdint>
  #include <random>
  #include <functional>
  #include <chrono>
  #include <mutex>
  #include <atomic>
#endif


#if defined(ARMA_USE_TBB_ALLOC)
  #include <tbb/scalable_allocator.h>
#endif


#if defined(ARMA_USE_MKL_ALLOC)
  #include <mkl_service.h>
#endif


#if !defined(ARMA_USE_CXX11)
  #if defined(ARMA_HAVE_TR1)
    #include <tr1/cmath>
    #include <tr1/complex>
  #endif
#endif


#include "armadillo_bits/include_atlas.hpp"
#include "armadillo_bits/include_hdf5.hpp"
#include "armadillo_bits/include_superlu.hpp"


#if defined(ARMA_USE_OPENMP)
  #include <omp.h>
#endif



//! \namespace arma namespace for Armadillo classes and functions
namespace arma
  {
  
  // preliminaries
  
  #include "armadillo_bits/arma_forward.hpp"
  #include "armadillo_bits/arma_static_check.hpp"
  #include "armadillo_bits/typedef_elem.hpp"
  #include "armadillo_bits/typedef_elem_check.hpp"
  #include "armadillo_bits/typedef_mat.hpp"
  #include "armadillo_bits/arma_str.hpp"
  #include "armadillo_bits/arma_version.hpp"
  #include "armadillo_bits/arma_config.hpp"
  #include "armadillo_bits/traits.hpp"
  #include "armadillo_bits/promote_type.hpp"
  #include "armadillo_bits/upgrade_val.hpp"
  #include "armadillo_bits/restrictors.hpp"
  #include "armadillo_bits/access.hpp"
  #include "armadillo_bits/span.hpp"
  #include "armadillo_bits/distr_param.hpp"
  #include "armadillo_bits/constants.hpp"
  #include "armadillo_bits/constants_old.hpp"
  #include "armadillo_bits/mp_misc.hpp"
  
  #ifdef ARMA_RNG_ALT
    #include ARMA_INCFILE_WRAP(ARMA_RNG_ALT)
  #else
    #include "armadillo_bits/arma_rng_cxx98.hpp"
  #endif
  
  #include "armadillo_bits/arma_rng_cxx11.hpp"
  #include "armadillo_bits/arma_rng.hpp"
  
  
  //
  // class prototypes
  
  #include "armadillo_bits/Base_bones.hpp"
  #include "armadillo_bits/BaseCube_bones.hpp"
  #include "armadillo_bits/SpBase_bones.hpp"
  
  #include "armadillo_bits/def_blas.hpp"
  #include "armadillo_bits/def_lapack.hpp"
  #include "armadillo_bits/def_atlas.hpp"
  #include "armadillo_bits/def_arpack.hpp"
  #include "armadillo_bits/def_superlu.hpp"
  #include "armadillo_bits/def_hdf5.hpp"
  
  #include "armadillo_bits/wrapper_blas.hpp"
  #include "armadillo_bits/wrapper_lapack.hpp"
  #include "armadillo_bits/wrapper_atlas.hpp"
  #include "armadillo_bits/wrapper_arpack.hpp"
  #include "armadillo_bits/wrapper_superlu.hpp"
  
  #include "armadillo_bits/cond_rel_bones.hpp"
  #include "armadillo_bits/arrayops_bones.hpp"
  #include "armadillo_bits/podarray_bones.hpp"
  #include "armadillo_bits/auxlib_bones.hpp"
  #include "armadillo_bits/sp_auxlib_bones.hpp"
  
  #include "armadillo_bits/injector_bones.hpp"
  
  #include "armadillo_bits/Mat_bones.hpp"
  #include "armadillo_bits/Col_bones.hpp"
  #include "armadillo_bits/Row_bones.hpp"
  #include "armadillo_bits/Cube_bones.hpp"
  #include "armadillo_bits/xvec_htrans_bones.hpp"
  #include "armadillo_bits/xtrans_mat_bones.hpp"
  #include "armadillo_bits/SizeMat_bones.hpp"
  #include "armadillo_bits/SizeCube_bones.hpp"
    
  #include "armadillo_bits/SpValProxy_bones.hpp"
  #include "armadillo_bits/SpMat_bones.hpp"
  #include "armadillo_bits/SpCol_bones.hpp"
  #include "armadillo_bits/SpRow_bones.hpp"
  #include "armadillo_bits/SpSubview_bones.hpp"
  #include "armadillo_bits/spdiagview_bones.hpp"
  #include "armadillo_bits/MapMat_bones.hpp"
  
  #include "armadillo_bits/typedef_mat_fixed.hpp"
  
  #include "armadillo_bits/field_bones.hpp"
  #include "armadillo_bits/subview_bones.hpp"
  #include "armadillo_bits/subview_elem1_bones.hpp"
  #include "armadillo_bits/subview_elem2_bones.hpp"
  #include "armadillo_bits/subview_field_bones.hpp"
  #include "armadillo_bits/subview_cube_bones.hpp"
  #include "armadillo_bits/diagview_bones.hpp"
  #include "armadillo_bits/subview_each_bones.hpp"
  #include "armadillo_bits/subview_cube_each_bones.hpp"
  #include "armadillo_bits/subview_cube_slices_bones.hpp"
  
  
  #include "armadillo_bits/diskio_bones.hpp"
  #include "armadillo_bits/wall_clock_bones.hpp"
  #include "armadillo_bits/running_stat_bones.hpp"
  #include "armadillo_bits/running_stat_vec_bones.hpp"
  
  #include "armadillo_bits/Op_bones.hpp"
  #include "armadillo_bits/OpCube_bones.hpp"
  #include "armadillo_bits/SpOp_bones.hpp"
  
  #include "armadillo_bits/eOp_bones.hpp"
  #include "armadillo_bits/eOpCube_bones.hpp"
  
  #include "armadillo_bits/mtOp_bones.hpp"
  #include "armadillo_bits/mtOpCube_bones.hpp"
  #include "armadillo_bits/mtSpOp_bones.hpp"
  
  #include "armadillo_bits/Glue_bones.hpp"
  #include "armadillo_bits/eGlue_bones.hpp"
  #include "armadillo_bits/mtGlue_bones.hpp"
  #include "armadillo_bits/SpGlue_bones.hpp"
  
  #include "armadillo_bits/GlueCube_bones.hpp"
  #include "armadillo_bits/eGlueCube_bones.hpp"
  #include "armadillo_bits/mtGlueCube_bones.hpp"
  
  #include "armadillo_bits/eop_core_bones.hpp"
  #include "armadillo_bits/eglue_core_bones.hpp"
  
  #include "armadillo_bits/GenSpecialiser.hpp"
  #include "armadillo_bits/Gen_bones.hpp"
  #include "armadillo_bits/GenCube_bones.hpp"
  
  #include "armadillo_bits/op_diagmat_bones.hpp"
  #include "armadillo_bits/op_diagvec_bones.hpp"
  #include "armadillo_bits/op_dot_bones.hpp"
  #include "armadillo_bits/op_inv_bones.hpp"
  #include "armadillo_bits/op_htrans_bones.hpp"
  #include "armadillo_bits/op_max_bones.hpp"
  #include "armadillo_bits/op_min_bones.hpp"
  #include "armadillo_bits/op_index_max_bones.hpp"
  #include "armadillo_bits/op_index_min_bones.hpp"
  #include "armadillo_bits/op_mean_bones.hpp"
  #include "armadillo_bits/op_median_bones.hpp"
  #include "armadillo_bits/op_sort_bones.hpp"
  #include "armadillo_bits/op_sort_index_bones.hpp"
  #include "armadillo_bits/op_sum_bones.hpp"
  #include "armadillo_bits/op_stddev_bones.hpp"
  #include "armadillo_bits/op_strans_bones.hpp"
  #include "armadillo_bits/op_var_bones.hpp"
  #include "armadillo_bits/op_repmat_bones.hpp"
  #include "armadillo_bits/op_repelem_bones.hpp"
  #include "armadillo_bits/op_reshape_bones.hpp"
  #include "armadillo_bits/op_vectorise_bones.hpp"
  #include "armadillo_bits/op_resize_bones.hpp"
  #include "armadillo_bits/op_cov_bones.hpp"
  #include "armadillo_bits/op_cor_bones.hpp"
  #include "armadillo_bits/op_shift_bones.hpp"
  #include "armadillo_bits/op_shuffle_bones.hpp"
  #include "armadillo_bits/op_prod_bones.hpp"
  #include "armadillo_bits/op_pinv_bones.hpp"
  #include "armadillo_bits/op_dotext_bones.hpp"
  #include "armadillo_bits/op_flip_bones.hpp"
  #include "armadillo_bits/op_reverse_bones.hpp"
  #include "armadillo_bits/op_princomp_bones.hpp"
  #include "armadillo_bits/op_misc_bones.hpp"
  #include "armadillo_bits/op_orth_null_bones.hpp"
  #include "armadillo_bits/op_relational_bones.hpp"
  #include "armadillo_bits/op_find_bones.hpp"
  #include "armadillo_bits/op_find_unique_bones.hpp"
  #include "armadillo_bits/op_chol_bones.hpp"
  #include "armadillo_bits/op_cx_scalar_bones.hpp"
  #include "armadillo_bits/op_trimat_bones.hpp"
  #include "armadillo_bits/op_cumsum_bones.hpp"
  #include "armadillo_bits/op_cumprod_bones.hpp"
  #include "armadillo_bits/op_symmat_bones.hpp"
  #include "armadillo_bits/op_hist_bones.hpp"
  #include "armadillo_bits/op_unique_bones.hpp"
  #include "armadillo_bits/op_toeplitz_bones.hpp"
  #include "armadillo_bits/op_fft_bones.hpp"
  #include "armadillo_bits/op_any_bones.hpp"
  #include "armadillo_bits/op_all_bones.hpp"
  #include "armadillo_bits/op_normalise_bones.hpp"
  #include "armadillo_bits/op_clamp_bones.hpp"
  #include "armadillo_bits/op_expmat_bones.hpp"
  #include "armadillo_bits/op_nonzeros_bones.hpp"
  #include "armadillo_bits/op_diff_bones.hpp"
  #include "armadillo_bits/op_norm_bones.hpp"
  #include "armadillo_bits/op_sqrtmat_bones.hpp"
  #include "armadillo_bits/op_logmat_bones.hpp"
  #include "armadillo_bits/op_range_bones.hpp"
  #include "armadillo_bits/op_chi2rnd_bones.hpp"
  #include "armadillo_bits/op_wishrnd_bones.hpp"
  #include "armadillo_bits/op_roots_bones.hpp"
  
  #include "armadillo_bits/glue_times_bones.hpp"
  #include "armadillo_bits/glue_mixed_bones.hpp"
  #include "armadillo_bits/glue_cov_bones.hpp"
  #include "armadillo_bits/glue_cor_bones.hpp"
  #include "armadillo_bits/glue_kron_bones.hpp"
  #include "armadillo_bits/glue_cross_bones.hpp"
  #include "armadillo_bits/glue_join_bones.hpp"
  #include "armadillo_bits/glue_relational_bones.hpp"
  #include "armadillo_bits/glue_solve_bones.hpp"
  #include "armadillo_bits/glue_conv_bones.hpp"
  #include "armadillo_bits/glue_toeplitz_bones.hpp"
  #include "armadillo_bits/glue_hist_bones.hpp"
  #include "armadillo_bits/glue_histc_bones.hpp"
  #include "armadillo_bits/glue_max_bones.hpp"
  #include "armadillo_bits/glue_min_bones.hpp"
  #include "armadillo_bits/glue_trapz_bones.hpp"
  #include "armadillo_bits/glue_atan2_bones.hpp"
  #include "armadillo_bits/glue_hypot_bones.hpp"
  #include "armadillo_bits/glue_polyfit_bones.hpp"
  #include "armadillo_bits/glue_polyval_bones.hpp"
  #include "armadillo_bits/glue_intersect_bones.hpp"
  #include "armadillo_bits/glue_affmul_bones.hpp"
  #include "armadillo_bits/glue_mvnrnd_bones.hpp"
  
  #include "armadillo_bits/gmm_misc_bones.hpp"
  #include "armadillo_bits/gmm_diag_bones.hpp"
  #include "armadillo_bits/gmm_full_bones.hpp"
  
  #include "armadillo_bits/spop_max_bones.hpp"
  #include "armadillo_bits/spop_min_bones.hpp"
  #include "armadillo_bits/spop_sum_bones.hpp"
  #include "armadillo_bits/spop_strans_bones.hpp"
  #include "armadillo_bits/spop_htrans_bones.hpp"
  #include "armadillo_bits/spop_misc_bones.hpp"
  #include "armadillo_bits/spop_diagmat_bones.hpp"
  #include "armadillo_bits/spop_mean_bones.hpp"
  #include "armadillo_bits/spop_var_bones.hpp"
  #include "armadillo_bits/spop_trimat_bones.hpp"
  #include "armadillo_bits/spop_symmat_bones.hpp"
  #include "armadillo_bits/spop_normalise_bones.hpp"
  #include "armadillo_bits/spop_reverse_bones.hpp"
  #include "armadillo_bits/spop_repmat_bones.hpp"
  
  #include "armadillo_bits/spglue_plus_bones.hpp"
  #include "armadillo_bits/spglue_minus_bones.hpp"
  #include "armadillo_bits/spglue_times_bones.hpp"
  #include "armadillo_bits/spglue_join_bones.hpp"
  #include "armadillo_bits/spglue_kron_bones.hpp"
  
  #if defined(ARMA_USE_NEWARP)
    #include "armadillo_bits/newarp_EigsSelect.hpp"
    #include "armadillo_bits/newarp_DenseGenMatProd_bones.hpp"
    #include "armadillo_bits/newarp_SparseGenMatProd_bones.hpp"
    #include "armadillo_bits/newarp_DoubleShiftQR_bones.hpp"
    #include "armadillo_bits/newarp_GenEigsSolver_bones.hpp"
    #include "armadillo_bits/newarp_SymEigsSolver_bones.hpp"
    #include "armadillo_bits/newarp_TridiagEigen_bones.hpp"
    #include "armadillo_bits/newarp_UpperHessenbergEigen_bones.hpp"
    #include "armadillo_bits/newarp_UpperHessenbergQR_bones.hpp"
  #endif
  
  
  //
  // low-level debugging and memory handling functions
  
  #include "armadillo_bits/debug.hpp"
  #include "armadillo_bits/memory.hpp"
  
  //
  // wrappers for various cmath functions
  
  #include "armadillo_bits/arma_cmath.hpp"
  
  //
  // classes that underlay metaprogramming 
  
  #include "armadillo_bits/unwrap.hpp"
  #include "armadillo_bits/unwrap_cube.hpp"
  #include "armadillo_bits/unwrap_spmat.hpp"
  
  #include "armadillo_bits/Proxy.hpp"
  #include "armadillo_bits/ProxyCube.hpp"
  #include "armadillo_bits/SpProxy.hpp"
  
  #include "armadillo_bits/diagmat_proxy.hpp"

  #include "armadillo_bits/strip.hpp"
  
  }
