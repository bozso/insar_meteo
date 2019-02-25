/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: statistics.h 502 2012-11-30 18:58:32Z cag $
 * ---------------------------------------------------------------------
 *
 * Copyright (C)  Niv Drory <drory@mpe.mpg.de>
 *                         Claus A. Goessl <cag@usm.uni-muenchen.de>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 *
 * ---------------------------------------------------------------------
 *
 */

#ifndef __LTL_MARRAY__
#error "<ltl/statistics.h> must be included after <ltl/marray.h>!"
#endif

#ifndef __LTL_STATISTICS__
#define __LTL_STATISTICS__

#include <ltl/config.h>

#include <algorithm> // partial_sort_copy method

using std::partial_sort_copy;

#include <ltl/marray/reductions.h>     // all reduction classes
#include <ltl/marray/partial_reduc.h>  // partial reductions
#include <ltl/marray/expr_iter.h>      // partial_sort_copy in median

namespace ltl {

/*! \addtogroup statistics
*/
//@{

/*! \defgroup stat_boolean Boolean valued reductions
*/

//@{

//---------------------------------------------------------------------
// ALL TRUE
//---------------------------------------------------------------------

/*! \defgroup stat_allof allof( Expr )

true if all elements of Expression are logically true.
*/
//@{
template<class Expr, int N>
bool allof( const ExprBase<Expr,N>& e )
{
#ifdef LTL_USE_SIMD
   allof_reduction<typename ExprNodeType<Expr>::expr_type::vec_value_type> R;
#else
   allof_reduction<typename Expr::value_type> R;
#endif
   typedef typename ExprNodeType<Expr>::expr_type ExprT;

   eval_full_reduction( ExprNode<ExprT,N>(ExprNodeType<Expr>::node(e.derived())), R );
   return R.result();
}
//@}

//---------------------------------------------------------------------
// NONE TRUE
//---------------------------------------------------------------------

/*! \defgroup stat_noneof noneof( Expr )

true if no element of Expression is logically true.
*/
//@{
template<class Expr, int N>
bool noneof( const ExprBase<Expr,N>& e )
{
#ifdef LTL_USE_SIMD
   noneof_reduction<typename ExprNodeType<Expr>::expr_type::vec_value_type> R;
#else
   noneof_reduction<typename Expr::value_type> R;
#endif
   typedef typename ExprNodeType<Expr>::expr_type ExprT;

   eval_full_reduction( ExprNode<ExprT,N>(ExprNodeType<Expr>::node(e.derived())), R );
   return R.result();
}
//@}

//---------------------------------------------------------------------
// AT LEAST 1 TRUE
//---------------------------------------------------------------------

/*! \defgroup stat_anyof anyof( Expr )

true if at least one element of Expression is logically true.
*/
//@{
template<class Expr, int N>
bool anyof( const ExprBase<Expr,N>& e )
{
#ifdef LTL_USE_SIMD
   anyof_reduction<typename ExprNodeType<Expr>::expr_type::vec_value_type> R;
#else
   anyof_reduction<typename Expr::value_type> R;
#endif
   typedef typename ExprNodeType<Expr>::expr_type ExprT;

   eval_full_reduction( ExprNode<ExprT,N>(ExprNodeType<Expr>::node(e.derived())), R );
   return R.result();
}
//@}

//@}

//---------------------------------------------------------------------
// COUNT LOGICAL TRUE
//---------------------------------------------------------------------

/*! \defgroup stat_count count( Expr )

Number of elements of Expression which are logically true.
*/
//@{
template<class Expr, int N>
int count( const ExprBase<Expr,N>& e )
{
   count_reduction<typename Expr::value_type> R;
   typedef typename ExprNodeType<Expr>::expr_type ExprT;

   eval_full_reduction( ExprNode<ExprT,N>(ExprNodeType<Expr>::node(e.derived())), R );
   return R.result();
}
//@}

/*! \defgroup stat_typet Type T valued reductions

*/

//@{

//---------------------------------------------------------------------
// MINIMUM VALUE
//---------------------------------------------------------------------

/*! \defgroup stat_min min( Expr [, T nan] )

Return minimum of Expression.
*/
//@{
template<class Expr, int N>
typename Expr::value_type min( const ExprBase<Expr,N>& e )
{
   min_reduction<typename Expr::value_type> R;
   typedef typename ExprNodeType<Expr>::expr_type ExprT;

   eval_full_reduction( ExprNode<ExprT,N>(ExprNodeType<Expr>::node(e.derived())), R );
   return R.result();
}

template<class Expr, int N>
typename Expr::value_type min( const ExprBase<Expr,N>& e, const typename Expr::value_type nan )
{
   min_reduction_nan<typename Expr::value_type> R(nan);
   typedef typename ExprNodeType<Expr>::expr_type ExprT;

   eval_full_reduction( ExprNode<ExprT,N>(ExprNodeType<Expr>::node(e.derived())), R );
   return R.result();
}
//@}

//---------------------------------------------------------------------
// MAXIMUM VALUE
//---------------------------------------------------------------------

/*! \defgroup stat_max max( Expr [, T nan] )

Return maximum of Expression.
*/
//@{
template<class Expr, int N>
typename Expr::value_type max( const ExprBase<Expr,N>& e )
{
   max_reduction<typename Expr::value_type> R;
   typedef typename ExprNodeType<Expr>::expr_type ExprT;

   eval_full_reduction( ExprNode<ExprT,N>(ExprNodeType<Expr>::node(e.derived())), R );
   return R.result();
}

template<class Expr, int N>
typename Expr::value_type max( const ExprBase<Expr,N>& e, const typename Expr::value_type nan )
{
   max_reduction_nan<typename Expr::value_type> R(nan);
   typedef typename ExprNodeType<Expr>::expr_type ExprT;

   eval_full_reduction( ExprNode<ExprT,N>(ExprNodeType<Expr>::node(e.derived())), R );
   return R.result();
}
//@}

//---------------------------------------------------------------------
// SUM
//---------------------------------------------------------------------

/*! \defgroup stat_sum sum( Expr [, T nan] )

Return sum over all elements of Expression.
*/
//@{
template<class Expr, int N>
typename Expr::value_type sum( const ExprBase<Expr,N>& e )
{
   sum_reduction<typename Expr::value_type> R;
   typedef typename ExprNodeType<Expr>::expr_type ExprT;

   eval_full_reduction( ExprNode<ExprT,N>(ExprNodeType<Expr>::node(e.derived())), R );
   return R.result();
}

template<class Expr, int N>
typename Expr::value_type sum( const ExprBase<Expr,N>& e, const typename Expr::value_type nan )
{
   sum_reduction_nan<typename Expr::value_type> R(nan);
   typedef typename ExprNodeType<Expr>::expr_type ExprT;

   eval_full_reduction( ExprNode<ExprT,N>(ExprNodeType<Expr>::node(e.derived())), R );
   return R.result();
}
//@}

//---------------------------------------------------------------------
// PRODUCT
//---------------------------------------------------------------------

/*! \defgroup stat_product product( Expr [, T nan] )

Return product of all elements of Expression.
*/
//@{
template<class Expr, int N>
typename Expr::value_type product( const ExprBase<Expr,N>& e )
{
   prod_reduction<typename Expr::value_type> R;
   typedef typename ExprNodeType<Expr>::expr_type ExprT;

   eval_full_reduction( ExprNode<ExprT,N>(ExprNodeType<Expr>::node(e.derived())), R );
   return R.result();
}

template<class Expr, int N>
typename Expr::value_type product( const ExprBase<Expr,N>& e, const typename Expr::value_type nan )
{
   prod_reduction_nan<typename Expr::value_type> R(nan);
   typedef typename ExprNodeType<Expr>::expr_type ExprT;

   eval_full_reduction( ExprNode<ExprT,N>(ExprNodeType<Expr>::node(e.derived())), R );
   return R.result();
}
//@}

//@}

/*! \defgroup stat_double Double valued reductions


*/

//@{

//---------------------------------------------------------------------
// AVERAGE
//---------------------------------------------------------------------

/*! \defgroup stat_average average( Expr [, T nan] )

Return mean value of \c Expr optionally neglecting <tt>values == nan</tt>.
*/
//@{
template<class Expr, int N>
double average( const ExprBase<Expr,N>& e )
{
   avg_reduction<typename Expr::value_type> R(e.derived().shape()->nelements());
   typedef typename ExprNodeType<Expr>::expr_type ExprT;

   eval_full_reduction( ExprNode<ExprT,N>(ExprNodeType<Expr>::node(e.derived())), R );
   return R.result();
}

template<class Expr, int N>
double average( const ExprBase<Expr,N>& e, const typename Expr::value_type nan )
{
   avg_reduction_nan<typename Expr::value_type> R(nan);
   typedef typename ExprNodeType<Expr>::expr_type ExprT;

   eval_full_reduction( ExprNode<ExprT,N>(ExprNodeType<Expr>::node(e.derived())), R );
   return R.result();
}
//@}

//---------------------------------------------------------------------
// VARIANCE (RMS**2)
//---------------------------------------------------------------------

/*! \defgroup stat_variance variance( Expr [, T nan] [, double* mean] )

Return variance of \c Expr optionally neglecting <tt>values == nan</tt> and
optionally store \c mean to pointed address.
*/
//@{
template<class Expr, int N>
double variance( const ExprBase<Expr,N>& e, double* const avg=NULL )
{
   const int n = e.derived().shape()->nelements();

   // first pass : get average
   const double aver = average(e);

   typedef typename ExprNodeType<Expr>::expr_type ExprT;
   ExprNode<ExprT,N> e2(ExprNodeType<Expr>::node(e.derived()));
   e2.reset();

   // second pass : calculate rms
   variance_reduction<typename Expr::value_type> R2( n, aver );
   eval_full_reduction( e2, R2 );

   if( avg != NULL )
      *avg = aver;

   return R2.result();
}

template<class Expr, int N>
double variance( const ExprBase<Expr,N>& e, const typename Expr::value_type nan,
                 double* const avg=NULL )
{
   // first pass : get average
   const double aver = average(e,nan);

   typedef typename ExprNodeType<Expr>::expr_type ExprT;
   ExprNode<ExprT,N> e2(ExprNodeType<Expr>::node(e.derived()));
   e2.reset();

   // second pass : calculate rms
   variance_reduction_nan<typename Expr::value_type> R2( aver, nan );
   eval_full_reduction( e2, R2 );

   if( avg != NULL )
      *avg = aver;

   return R2.result();
}
//@}

//---------------------------------------------------------------------
// STANDARD DEVIATION
//---------------------------------------------------------------------

/*! \defgroup stat_stddev stddev( Expr [, T nan] [, double* mean] )

Return standard deviation of \c Expr optionally neglecting
<tt>values == nan</tt> and optionally store \c mean to pointed address.
<tt>// = sqrt( variance(...) )</tt>
*/
//@{
template<class Expr, int N>
double stddev( const ExprBase<Expr,N>& e, double* avg=NULL )
{
   return ::sqrt( variance( e, avg ) );
}

template<class Expr, int N>
double stddev( const ExprBase<Expr,N>& e, typename Expr::value_type nan,
               double* avg=NULL )
{
   const double var = variance( e, nan, avg );
   if(var != double(nan))
      return ::sqrt(var);
   return var;
}

//@}

//---------------------------------------------------------------------
// MEDIAN
//---------------------------------------------------------------------

/*! \defgroup stat_median_exact median_exact( Expr [, T nan] )

Return median value of \c Expr optionally neglecting <tt>values == nan</tt>.
*/
//@{
template<class Expr, int N>
double median_exact( const ExprBase<Expr,N>& a )
{
   typedef typename Expr::value_type T;
   Expr& e = const_cast<Expr&>(a.derived());

   const int marray_length =  e.shape()->nelements();
   const bool is_odd = marray_length % 2;
   const int array_length = (marray_length / 2) + 1;
   MArray<T,1> array_(array_length);

   partial_sort_copy(e.begin(), e.end(), array_.beginRA(), array_.endRA());

   if (is_odd)
      return double(array_(array_length));
   else
      return double( (array_(array_length - 1) + array_(array_length)) / 2.0 );
}


// help class for exact median with nan
template<class T>
class less_nan// : public binary_function<T, T, T>
{
   public:
      less_nan(const T nan)
            : nan_(nan)
      { }
      bool operator()( const T i, const T j ) const
      {
         if( i == nan_ )
            return false;
         if( j == nan_ )
            return true;
         return ( i < j );
      }
   private:
      const T nan_;
};


/*! \warning This Overload of \c median_exact() may be a high-cost.
  Expression is evaluated twice.
*/
template<class Expr, int N>
double median_exact( const ExprBase<Expr,N>& a, const typename Expr::value_type nan )
{
   typedef typename Expr::value_type T;
   Expr& e = const_cast<Expr&>(a.derived());

   const int marray_length = count(e != nan);
   if(marray_length == 0)
      return double(nan);
   const bool is_odd = marray_length % 2;
   const int array_length = (marray_length / 2) + 1;
   MArray<T,1> array_(array_length);

   partial_sort_copy(e.begin(), e.end(), array_.beginRA(), array_.endRA(), less_nan<T>(nan));

   if (is_odd)
      return double(array_(array_length));
   else
      return double( (array_(array_length - 1) + array_(array_length)) / 2.0 );
}
//@}

/*! \defgroup stat_median_estimate median_estimate( Expr, int bins, T min, T max [, T nan] )

Return mode of a histogram of \c Expr (see \ref stat_histogram below).
*/
//@{
//! Helper class for \c median_estimate().
/*! Return mode of a histogram of bins \e bins, starting at \e min,
with step size \e step (see \ref stat_histogram below).
(Exactly: left boundary of bin holding the bisector of the histogram)
*/
template<class T>
double mode_histogram( const MArray<T, 1>& h, const double min,
                       const double step, const int sum )
{
   const bool is_odd = sum % 2;
   const int half_size = sum / 2;

   int count = 0;
   int i = 0;

   while( count <= half_size )
   {
      ++i;
      count += h(i);
   }

   if( is_odd || ((count - 1) > half_size) )
      return ( min + (double(i-1) * step) );

   int j = i-1;
   while( (h(j) == 0) && (j>1) )
      --j;
   return (min + ( double(i+j-2) * step / 2.0 ) );
}


template<class Expr, int N>
double median_estimate( const ExprBase<Expr,N>& e, const int bins,
                        const typename Expr::value_type min,
                        const typename Expr::value_type max,
                        double * stepptr = NULL, int * sumptr = NULL)
{
   double step;
   int sum;
   const MArray<int,1> h( histogram( e, bins, min, max, &step, &sum ) );
   if(stepptr) *stepptr = step;
   if(sumptr) *sumptr = sum;
   return ( mode_histogram( h, double(min), step, sum) );
}


template<class Expr, int N>
double median_estimate( const ExprBase<Expr,N>& e, const int bins,
                        const typename Expr::value_type min,
                        const typename Expr::value_type max,
                        const typename Expr::value_type nan,
                        double * stepptr = NULL, int * sumptr = NULL)
{
   double step;
   int sum;
   const MArray<int,1> h( histogram( e, bins, min, max, nan, &step, &sum ) );
   if(stepptr) *stepptr = step;
   if(sumptr) *sumptr = sum;
   return ( mode_histogram( h, double(min), step, sum) );
}
//@}

//@}

//---------------------------------------------------------------------
// HISTOGRAM
// min incl., max excl., range = [min, max)
// one bin = [ (max - min) / nobins )
//---------------------------------------------------------------------

/*! \defgroup stat_histogram histogram ( Expr, int bins, T min, T max [, T nan] )

Return the histogram of \c Expr as an ltl::MArray<T, 1>, build \c bins bins
in the range <tt>[min, max)</tt>, in every bin include left boundary,
exclude right, optionally neglect <tt>values == nan</tt>.
*/
//@{
/*
template<class T,int N>
MArray<int, 1> histogram( const MArray<T,N>& a, const int bins,
                          const T min, const T max,
                          double* const step = NULL, int* const sum = NULL)
{
   ExprNode<typename MArray<T,N>::const_iterator,N> e( a.begin() );
   return histogram(e, bins, min, max, step, sum);
}
*/

template<class Expr, int N>
MArray<int,1> histogram( const ExprBase<Expr,N>& a, const int bins,
                         const typename Expr::value_type min,
                         const typename Expr::value_type max,
                         double* const step = NULL, int* const sum = NULL)
{
   histogram_reduction<typename Expr::value_type> R(bins, min, max);
   typedef typename ExprNodeType<Expr>::expr_type ExprT;
   const ExprNode<ExprT,N>& e(ExprNodeType<Expr>::node(a.derived()));
   eval_full_reduction( e, R );
   if (step != NULL)
      *step = R.step();
   if (sum != NULL)
      *sum = R.sum();
   return R.result();
}

template<class Expr, int N>
MArray<int,1> histogram( const ExprBase<Expr,N>& a, const int bins,
                         const typename Expr::value_type min,
                         const typename Expr::value_type max,
                         const typename Expr::value_type nan,
                         double* const step = NULL, int* const sum = NULL)
{
   histogram_reduction_nan<typename Expr::value_type> R(bins, min, max, nan);
   typedef typename ExprNodeType<Expr>::expr_type ExprT;
   const ExprNode<ExprT,N>& e(ExprNodeType<Expr>::node(a.derived()));
   eval_full_reduction( e, R );
   if (step != NULL)
      *step = R.step();
   if (sum != NULL)
      *sum = R.sum();
   return R.result();
}
//@}


//---------------------------------------------------------------------
// BIWEIGHT MEAN AND DISPERSION
//---------------------------------------------------------------------

/*! \defgroup biweight Robust mean and dispersion using Tukey's biweight

*/
//@{

/*!
Calculate a resistant estimate of the dispersion of a distribution.
For an uncontaminated distribution, this is identical to the standard deviation.
Use the median absolute deviation as the initial estimate, then weight
points using Tukey's Biweight. See, for example, "Understanding Robust
and Exploratory Data Analysis," by Hoaglin, Mosteller and Tukey, John Wiley & Sons, 1983.

If \c zero is \c true, the dispersion is caluculated around
the value 0 instead of the average value. For example, if \c Expr is a vector
of residuals, \c zero should be set to \c true.

Returns the dispersion, -1 in case of error.
*/
template<class Expr, int N>
typename Expr::value_type robust_sigma (const ExprBase<Expr,N>& a, const bool zero = false)
{
   Expr& y = const_cast<Expr&>(a.derived());
   typedef typename Expr::value_type T;

   const T eps = numeric_limits<T>::epsilon()*10.0;
   const T y0 = (zero ? T (0) : median_exact (y));

   // First, the median absolute deviation about the median:
   T mad = median_exact (fabs (y-y0))/0.6745;

   // If the MAD=0, try the MEAN absolute deviation:
   if (mad<eps)
      mad = average (fabs (y-y0))/0.8;
   if (mad<eps)
      return T (0);

   // Now the biweighted value:
   MArray<T,N> uu = pow2 ((y-y0)/(6.*mad));

   const int n = y.shape()->nelements();

   const int cn = count (uu<1.0);
   if (cn<3)
      return T(-1);
   const T numerator = sum (merge (uu<1.0, pow2 (y-y0)*pow4 (1.0-uu), 0.0));
   const T den = sum (merge (uu<1.0, (1.0-uu)*(1.0-5.0*uu), 0.0));

   const T sigma = n*numerator/(den*(den-1));
   if (sigma>0.0)
      return std::sqrt(sigma);
   else
      return T (0);
}

/*!
   Calculate the center and dispersion (like mean and sigma) of a
   distribution using bisquare weighting (Tukey's bisquare). Optionally
   control the maximum number of iterations in the parameter \c maxit.

   Return the mean, and optionally the dispersion (stddev) in \c *rsigma.
*/
template<class Expr, int N>
typename Expr::value_type biweight_mean (const ExprBase<Expr,N>& a, typename Expr::value_type* rsigma=NULL, const int maxit=20)
{
   Expr& y = const_cast<Expr&>(a.derived());
   typedef typename Expr::value_type T;

   const T eps = numeric_limits<T>::epsilon()*10.0;
   const int n = y.shape ()->nelements ();
   const T close_enough = 0.03*std::sqrt (0.5/(n-1.0)); // compare to fractional change in width

   T diff = numeric_limits<T>::max();
   int itnum = 0;

   // As an initial estimate of the center, use the median:
   T y0 = median_exact (y);

   // Calculate the weights:
   MArray<T,N> dev = y-y0;
   T sigma = robust_sigma (dev), prev_sigma = 0;
   if (sigma<eps) // the median is it.
   {
      if (rsigma)
         *rsigma = sigma;
      return y0;
   }

   // now, iterate
   while ((diff>close_enough) && (++itnum<maxit))
   {
      MArray<T,N> uu = pow2 ((y-y0)/(6.*sigma));
      uu = merge (uu<1.0, uu, 1.0);
      MArray<T,N> weights = pow2 (1.-uu);
      weights = weights/sum (weights);
      y0 = sum (weights*y);
      dev = y-y0;
      prev_sigma = sigma;
      sigma = robust_sigma (dev, true);
      if (sigma>eps)
         diff = std::fabs (prev_sigma-sigma)/prev_sigma;
      else
         diff = 0.0;
   }
   if (rsigma)
      *rsigma = sigma;
   return y0;
}

//@}

/*! \defgroup stat_kappasigma Kappa-Sigma Clipping

 */
//@{

//---------------------------------------------------------------------
// KAPPA-SIGMA CLIPPED MEAN AND VARIANCE
//---------------------------------------------------------------------

/*! \defgroup stat_kappa_sigma_average kappa_sigma_average( Expr, double kappa, [T nan,] double* mean [, double* sigma] )

  Return number of elements after clipping and
  store \f$\kappa\sigma\f$ clipped mean (and optionally resulting \c sigma) to
  pointed address, also optionally neglecting <tt>values == nan</tt>.
*/
//@{
template<class Expr, int N>
int kappa_sigma_average( const ExprBase<Expr,N>& a, const double kappa,
                         double* const mean, double* const sigma = NULL)
{
   typedef typename ExprNodeType<Expr>::expr_type ExprT;

   // initial guess
   double avg, var, stddev;
   var = variance( a, &avg );
   stddev = ::sqrt(var);

   ExprNode<ExprT,N> e(ExprNodeType<Expr>::node(a.derived()));
   int newn = e.shape()->nelements();
   int n = 0;

   while( (newn != n) && (var != 0.0) )
   {

      n = newn;
      newn = 0;

      stddev *= kappa;

      // new mean
      e.reset();
      kappa_mean_reduction<typename Expr::value_type> R1( stddev, avg );
      eval_full_reduction( e, R1 );
      avg = R1.result();

      // new sigma
      e.reset();
      kappa_sigma_reduction<typename Expr::value_type> R2( stddev, avg );
      eval_full_reduction( e, R2 );
      var = R2.result();
      stddev = ::sqrt(var);
      newn = R2.newn();
   }

   if( sigma != NULL )
      *sigma = stddev;
   *mean  = avg;
   return newn;
}

template<class Expr, int N>
int kappa_sigma_average( const ExprBase<Expr,N>& a, const double kappa,
                         const typename Expr::value_type nan,
                         double* const mean, double* const sigma = NULL)
{
   typedef typename ExprNodeType<Expr>::expr_type ExprT;

   // initial guess
   double avg, var, stddev;
   var = variance( a, nan, &avg );
   if(var != double(nan))
      stddev = ::sqrt(var);
   else stddev = var;

   ExprNode<ExprT,N> e(ExprNodeType<Expr>::node(a.derived()));
   int newn = e.shape()->nelements();
   int n = 0;

   while( (newn != n) && (var != 0.0) && (var != double(nan)) )
   {
      n = newn;
      newn = 0;

      stddev *= kappa;

      // new mean
      e.reset();
      kappa_mean_reduction_nan<typename Expr::value_type> R1( stddev, avg, nan );
      eval_full_reduction( e, R1 );
      avg = R1.result();

      // new sigma
      e.reset();
      kappa_sigma_reduction_nan<typename Expr::value_type> R2( stddev, avg, nan );
      eval_full_reduction( e, R2 );
      var = R2.result();
      if(var != double(nan))
         stddev = ::sqrt(var);
      else stddev = var;
      newn = R2.newn();
   }

   if( sigma != NULL )
   {
      *sigma = stddev;
   }

   *mean  = avg;
   return newn;
}
//@}

//---------------------------------------------------------------------
// KAPPA-SIGMA CLIPPED MEDIAN
//---------------------------------------------------------------------

/*! \defgroup stat_kappa_sigma_median kappa_sigma_median( Expr, double kappa, [T nan,] double* mean [, double* sigma] )

Return number of elements after clipping and
store \f$\kappa\sigma\f$ clipped median (and optionally resulting \c sigma) to
pointed address, also optionally neglecting <tt>values == nan</tt>.
 */
//@{
//! Return mode of sorted ltl::MArray<T, 1>.
template<class T>
int median_sorted_array( const MArray<T, 1>& array_, const double kappa,
                         double* const median, double* const sigma = NULL)
{
   int newn = array_.nelements();
   int n = 0;

   // initial guess
   double med, var, stddev;
   const int half_a = newn / 2;
   med = double( array_( half_a + 1 ) );
   if( !(newn % 2) )
   {
      med += double( array_( half_a ) );
      med /= 2.0;
   }
   ExprNode<typename MArray<T, 1>::const_iterator, 1> e( array_.begin() );
   variance_reduction<T> R( newn, med );
   eval_full_reduction( e, R );
   var = R.result();
   stddev = ::sqrt(var);

   while( (newn != n) && (var != 0.0) )
   {
      n = newn;
      newn = 0;
      stddev *= kappa;

      // new median
      e.reset();
      kappa_median_reduction<T> R1( array_, med - stddev, med + stddev);
      eval_full_reduction( e, R1 );
      med = R1.result();

      // new sigma
      e.reset();
      kappa_sigma_reduction<T> R2( stddev, med );
      eval_full_reduction( e, R2 );
      var = R2.result();
      stddev = ::sqrt(var);
      newn = R2.newn();
   }

   if( sigma != NULL )
      *sigma = stddev;
   *median  = med;
   return newn;
}


template<class Expr, int N>
int kappa_sigma_median( const ExprBase<Expr,N>& e, const double kappa,
                        double* const median, double* const sigma = NULL)
{
   // it's worth sorting once and clipping afterwards
   Expr& E = const_cast<Expr&>(e.derived());
   MArray<typename Expr::value_type,1> array_( E.shape()->nelements() );
   partial_sort_copy(E.begin(), E.end(), array_.beginRA(), array_.endRA());

   return median_sorted_array( array_, kappa, median, sigma );
}

template<class Expr, int N>
int kappa_sigma_median( const ExprBase<Expr,N>& e, const double kappa,
                        const typename Expr::value_type nan,
                        double* const median, double* const sigma = NULL)
{
   Expr& E = const_cast<Expr&>(e.derived());
   int array_len = count(E!=nan);
   if(array_len == 0)
   {
      *median = double(nan);
      if(sigma != NULL)
         *sigma = double(nan);
      return 0;
   }
   MArray<typename Expr::value_type,1> array_( array_len );
   partial_sort_copy(E.begin(), E.end(), array_.beginRA(), array_.endRA(),
                     less_nan<typename Expr::value_type>(nan));

   return median_sorted_array( array_, kappa, median, sigma);
}
//@}

//---------------------------------------------------------------------
// KAPPA-SIGMA MEDIAN CLIPPED AVERAGE
//---------------------------------------------------------------------

/*! \defgroup stat_kappa_median_average kappa_median_average( Expr, double kappa, [T nan,] double* mean [, double* sigma] )

Clip according to median, but store average of remaining to \c mean.
Return number of elements after clipping (and optionally resulting \c sigma
in pointed address.
Optionally neglect <tt>values == nan</tt>.
 */
//@{
//! Return average after median clipping sorted ltl::MArray<T, 1>
template<class T>
int median_clip_average( const MArray<T, 1>& a, const double kappa,
                         double* const mean, double* const sigma = NULL)
{

   double locsigma = 0.0;

   // median clip sorted MArray
   const int n = median_sorted_array( a, kappa, mean, &locsigma);

   // get boundaries
   const double ks = kappa * locsigma;
   const T lower_limit = T( *mean - ks );
   const T upper_limit = T( *mean + ks );

   int i = 1;
   typename MArray<T, 1>::const_iterator a_iter = a.begin();

   while( (!a_iter.done()) && (*a_iter < lower_limit) )
   {
      ++a_iter;
      ++i;
   }
   const int lower_index = i;
   while( (!a_iter.done()) && (*a_iter <= upper_limit) )
   {
      ++a_iter;
      ++i;
   }
   if( i == lower_index )
      ++i;
   const int upper_index = i - 1;

#ifdef LTL_RANGE_CHECKING
   // check consistency of clipping
   if(n != (1 + upper_index - lower_index))
      throw RangeException("inconsistent clipping in median_clip_average");
#endif

   // get average and stddev of clipped MArray
   if(sigma != NULL)
      *sigma = stddev( a( Range(lower_index, upper_index) ), mean );
   else // or just average
      *mean = average( a( Range(lower_index, upper_index) ) );

   return n;
}

template<class Expr, int N>
int kappa_median_average( const ExprBase<Expr,N>& e, const double kappa,
                          double* const mean, double* const sigma = NULL)
{
   // it's worth sorting once and clipping afterwards
   Expr& E = const_cast<Expr&>(e.derived());
   MArray<typename Expr::value_type,1> array_( E.shape()->nelements() );
   partial_sort_copy(E.begin(), E.end(), array_.beginRA(), array_.endRA());

   return median_clip_average( array_, kappa, mean, sigma );
}

template<class Expr, int N>
int kappa_median_average( const ExprBase<Expr,N>& e, const double kappa,
                          const typename Expr::value_type nan,
                          double* const mean, double* const sigma = NULL)
{
   Expr& E = const_cast<Expr&>(e.derived());
   const int array_len = count(E!=nan);
   if(array_len == 0)
   {
      *mean = double(nan);
      if(sigma != NULL)
         *sigma = double(nan);
      return 0;
   }
   MArray<typename Expr::value_type,1> array_( array_len );
   partial_sort_copy(E.begin(), E.end(), array_.beginRA(), array_.endRA(),
                     less_nan<typename Expr::value_type>(nan));

   return median_clip_average( array_, kappa, mean, sigma);
}
//@}

//---------------------------------------------------------------------
// KAPPA-SIGMA AVERAGE CLIPPED MEDIAN
//---------------------------------------------------------------------

/*! \defgroup stat_kappa_average_median kappa_average_median( Expr, double kappa, [T nan,] double* mean [, double* sigma] )

Clip according to average, but store median of remaining to \c mean.
Return number of elements after clipping (and optionally resulting \c sigma
in pointed address.
Optionally neglect <tt>values == nan</tt>.
 */
//@{
//! Return median after average clipping sorted ltl::MArray<T, 1>
template<class T>
int average_clip_median( const MArray<T, 1>& a, const double kappa,
                         double* const mean, double* const sigma = NULL)
{
   double locsigma = 0.0;

   // average clip sorted MArray
#ifdef LTL_RANGE_CHECKING
   const int n = kappa_sigma_average( a, kappa, mean, &locsigma);
#else
   kappa_sigma_average( a, kappa, mean, &locsigma);
#endif

   // get boundaries
   const double ks = kappa * locsigma;
   const T lower_limit = T( *mean - ks );
   const T upper_limit = T( *mean + ks );

   int i = 1;
   typename MArray<T, 1>::const_iterator a_iter = a.begin();

   while( (!a_iter.done()) && (*a_iter < lower_limit) )
   {
      ++a_iter;
      ++i;
   }
   const int lower_index = i;
   while( (!a_iter.done()) && (*a_iter <= upper_limit) )
   {
      ++a_iter;
      ++i;
   }
   if( i == lower_index )
      ++i;
   const int upper_index = i - 1;

   // get median index
   const int total_length = 1 + upper_index - lower_index;
   const bool is_odd = total_length % 2;
   const int half_length = (total_length / 2) + lower_index;

#ifdef LTL_RANGE_CHECKING
   // check consistency of clipping
   if(n != total_length)
      throw RangeException("inconsistent clipping in average_clip_median");
#endif

   // get median within clipped range
   if (is_odd)
      *mean = double( a(half_length) );
   else
      *mean = double( (a(half_length - 1) + a(half_length)) / T(2) );

   // calculate stddev
   if(sigma != NULL)
   {
      ExprNode<typename MArray<T,1>::const_iterator,1>
      e( (a( Range(lower_index, upper_index) )).begin() );
      variance_reduction<T> R( total_length, *mean );
      eval_full_reduction( e, R );
      *sigma = ::sqrt(R.result());
   }

   // return size of clipped array
   return total_length;
}


template<class Expr, int N>
int kappa_average_median( const ExprBase<Expr,N>& e, const double kappa,
                          double* const mean, double* const sigma = NULL)
{
   // it's worth sorting once and clipping afterwards
   Expr& E = const_cast<Expr&>(e.derived());
   MArray<typename Expr::value_type, 1> array_( E.shape()->nelements() );
   partial_sort_copy(E.begin(), E.end(), array_.beginRA(), array_.endRA());

   return average_clip_median( array_, kappa, mean, sigma );
}


template<class Expr, int N>
int kappa_average_median( const ExprBase<Expr,N>& e, const double kappa,
                          const typename Expr::value_type nan,
                          double* const mean, double* const sigma = NULL)
{
   Expr& E = const_cast<Expr&>(e.derived());

   const int array_len = count(E!=nan);
   if(array_len == 0)
   {
      *mean = double(nan);
      if(sigma != NULL)
         *sigma = double(nan);
      return 0;
   }
   MArray<typename Expr::value_type,1> array_( array_len );
   partial_sort_copy(E.begin(), E.end(), array_.beginRA(), array_.endRA(),
                     less_nan<typename Expr::value_type>(nan));

   return average_clip_median( array_, kappa, mean, sigma);
}
//@}

//@}

//@}

}

#endif
