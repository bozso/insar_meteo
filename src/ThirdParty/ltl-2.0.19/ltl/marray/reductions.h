/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: reductions.h 491 2011-09-02 19:36:39Z drory $
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


#ifndef  __LTL_STATISTICS__
#error "<ltl/marray/reductions.h> must be included via <ltl/statistics.h>, never alone!"
#endif


#ifndef __LTL_REDUCTIONS__
#define __LTL_REDUCTIONS__

#ifdef HAVE_NUMERIC_LIMITS
#  include <limits>
#else
#  include <ltl/misc/limits_hack.h>
#endif

#include <ltl/marray/eval_reduc.h>

#ifdef HAVE_NUMERIC_LIMITS
using std::numeric_limits;
#else
using ltl::numeric_limits;
#endif

namespace ltl {

// forward declarations of vectorized versions
// include<> at end of this file.
#if defined(LTL_USE_SIMD) && defined(__SSE2__)
template<typename T> class sum_reduction_vec;
template<typename T> class avg_reduction_vec;
template<typename T> class avg_reduction_nan_vec;
template<typename T> class variance_reduction_vec;
template<typename T> class variance_reduction_nan_vec;
template<typename T> class sum_reduction_nan_vec;
template<typename T> class min_reduction_vec;
template<typename T> class max_reduction_vec;
template<typename T> class count_reduction_vec;
template<typename T> class allof_reduction_vec;
template<typename T> class noneof_reduction_vec;
template<typename T> class anyof_reduction_vec;
#endif

      //! Just for class browsers ...
class _reduction_base
{
   public:
   enum { isVectorizable = 0 };
};

/*! \class _reduction_base
  The reduction operation classes define two methods, \n
  bool evaluate( T x );   // evaluate one step for element x, and \n
  whatever result() const // return final result \n

  evaluate() returns bool so that the evaluation of the expression
  can be stopped if the result is known before all elements
  have been evaluated, e.g. for anyof() or allof().
  evaluate() has to return false if evaluation should stop.
*/

//---------------------------------------------------------------------
// SUM
//---------------------------------------------------------------------

template<class T>
class sum_reduction : public _reduction_base
{
   public:
      sum_reduction()
            : sum_(0)
      {  }

      inline bool evaluate( const T x )
      {
         sum_ += x;
         return true;
      }

      inline T result() const
      {
         return sum_;
      }

      inline void copyResult(const sum_reduction<T>& other)
      {
         sum_ = other.sum_;
      }

      enum { isVectorizable = 1 };
      typedef T value_type;
#ifdef LTL_USE_SIMD
      typedef T vec_calc_type;
      friend class sum_reduction_vec<T>;
      typedef sum_reduction_vec<T> vec_reduction;
      inline void copyResult( const vec_reduction& other )
      {
         sum_ = other.result();
      }
#endif

   private:
      T sum_;
};

template<class T>
class sum_reduction_nan : public _reduction_base
{
   public:
      sum_reduction_nan(const T nan)
            : sum_(T(0)), nan_(nan), n_(0)
      {  }

      inline bool evaluate( const T x )
      {
         if(x != nan_)
         {
            sum_ += x;
            ++n_;
         }
         return true;
      }

      inline T result() const
      {
         return sum_;
      }

      inline int n() const
      {
         return n_;
      }

      inline void copyResult(const sum_reduction_nan<T>& other)
      {
         sum_ = other.sum_;
         n_   = other.n_;
      }

      enum { isVectorizable = 1 };
      typedef T value_type;
#ifdef LTL_USE_SIMD
      typedef T vec_calc_type;
      friend class sum_reduction_nan_vec<T>;
      typedef sum_reduction_nan_vec<T> vec_reduction;
      inline void copyResult( const vec_reduction& other )
      {
         sum_ = other.result();
         n_ = other.n();
      }
#endif

   private:
      T sum_;
      const T nan_;
      int n_;
};


//---------------------------------------------------------------------
// PRODUCT
//---------------------------------------------------------------------

template<class T>
class prod_reduction : public _reduction_base
{
   public:
      prod_reduction()
            : prod_(1)
      {  }

      inline bool evaluate( const T x )
      {
         prod_ *= x;
         return true;
      }

      inline T result() const
      {
         return prod_;
      }

      inline void copyResult(const prod_reduction<T>& other)
      {
         prod_ = other.prod_;
      }

      enum { isVectorizable = 0 };
      typedef T value_type;

   private:
      T prod_;
};

template<class T>
class prod_reduction_nan : public _reduction_base
{
   public:
      prod_reduction_nan(const T nan)
            : prod_(1), nan_(nan), n_(0)
      {  }

      inline bool evaluate( const T x )
      {
         if (x != nan_)
         {
            prod_ *= x;
            ++n_;
         }
         return true;
      }

      inline T result() const
      {
         return prod_;
      }

      inline int n() const
      {
         return n_;
      }

      inline void copyResult(const prod_reduction_nan<T>& other)
      {
         prod_ = other.prod_;
         n_    = other.n_;
      }

      enum { isVectorizable = 0 };
      typedef T value_type;

   private:
      T prod_;
      const T nan_;
      int n_;
};


//---------------------------------------------------------------------
// AVERAGE
//---------------------------------------------------------------------

template<class T>
class avg_reduction : public _reduction_base
{
   public:
      avg_reduction( const int n )
            : sum_(0.0), n_(n)
      {  }

      inline bool evaluate( T x )
      {
         sum_ += double(x);
         return true;
      }

      inline double result() const
      {
         return sum_ / double(n_);
      }

      inline double sum() const
      {
         return sum_;
      }

      inline int n() const
      {
         return n_;
      }

      inline void copyResult(const avg_reduction<T>& other)
      {
         sum_ = other.sum_;
      }

      enum { isVectorizable = 1 };
      typedef double value_type;
#ifdef LTL_USE_SIMD
      typedef T vec_calc_type;
      friend class avg_reduction_vec<T>;
      typedef avg_reduction_vec<T> vec_reduction;
      inline void copyResult( const vec_reduction& other )
      {
         sum_ = other.sum();
      }
#endif

   private:
      double sum_;
      const int n_;
};

template<class T>
class avg_reduction_nan : public _reduction_base
{
   public:
      avg_reduction_nan( const T nan )
            : sum_(0.0), nan_(nan), n_(0)
      {  }

      inline bool evaluate( const T x )
      {
         if(x != nan_)
         {
            sum_ += double(x);
            ++n_;
         }
         return true;
      }

      inline double result() const
      {
         if(n_ > 0)
            return sum_ / double(n_);
         return double(nan_);
      }

      inline double sum() const
      {
         return sum_;
      }

      inline int n() const
      {
         return n_;
      }

      inline void copyResult(const avg_reduction_nan<T>& other)
      {
         sum_ = other.sum_;
         n_   = other.n_;
      }

      enum { isVectorizable = 1 };
      typedef double value_type;
#ifdef LTL_USE_SIMD
      typedef T vec_calc_type;
      friend class avg_reduction_nan_vec<T>;
      typedef avg_reduction_nan_vec<T> vec_reduction;
      inline void copyResult( const vec_reduction& other )
      {
         sum_ = other.sum();
         n_ = other.n();
      }
#endif

   private:
      double sum_;
      const T nan_;
      int n_;
};


//---------------------------------------------------------------------
// VARIANCE (RMS**2)
//---------------------------------------------------------------------

template<class T>
class variance_reduction : public _reduction_base
{
   public:
      variance_reduction( const int n, const double avg )
            : sum_(0.0), n_(n), avg_(avg)
      {  }

      inline bool evaluate( const T x )
      {
         const double tmp = double(x) - avg_;
         sum_ += tmp * tmp;
         return true;
      }

      inline double result() const
      {
         if(n_ > 1)
            return sum_ / double(n_ - 1);
         return 0.0;
      }

      inline double sum() const
      {
         return sum_;
      }

      inline int n() const
      {
         return n_;
      }

      inline void copyResult(const variance_reduction<T>& other)
      {
         sum_ = other.sum_;
      }

      enum { isVectorizable = 1 };
      typedef double value_type;
#ifdef LTL_USE_SIMD
      typedef T vec_calc_type;
      friend class variance_reduction_vec<T>;
      typedef variance_reduction_vec<T> vec_reduction;
      inline void copyResult( const vec_reduction& other )
      {
         sum_ = other.sum();
      }
#endif

   private:
      double sum_;
      const int n_;
      const double avg_;
};

template<class T>
class variance_reduction_nan : public _reduction_base
{
   public:
      variance_reduction_nan( const double avg, const T nan )
            : sum_(0.0), n_(0), avg_(avg), nan_(nan)
      {  }

      inline bool evaluate( const T x )
      {
         if(x != nan_)
         {
            const double tmp = double(x) - avg_;
            sum_ += tmp * tmp;
            ++n_;
         }
         return true;
      }

      inline double result() const
      {
         if(n_ > 1)
            return sum_ / double(n_ - 1);
         if(n_ == 1)
            return 0.0;
         return double(nan_);
      }

      inline double sum() const
      {
         return sum_;
      }

      inline int n() const
      {
         return n_;
      }

      inline void copyResult(const variance_reduction_nan<T>& other)
      {
         sum_ = other.sum_;
         n_   = other.n_;
      }

      enum { isVectorizable = 1 };
      typedef double value_type;
#ifdef LTL_USE_SIMD
      typedef T vec_calc_type;
      friend class variance_reduction_nan_vec<T>;
      typedef variance_reduction_nan_vec<T> vec_reduction;
      inline void copyResult( const vec_reduction& other )
      {
         sum_ = other.sum();
         n_ = other.n();
      }
#endif

   private:
      double sum_;
      int n_;
      const double avg_;
      const T nan_;
};


//---------------------------------------------------------------------
// KAPPA-SIGMA CLIPPED MEAN AND VARIANCE
//---------------------------------------------------------------------

template<class T>
class kappa_mean_reduction : public _reduction_base
{
   public:
      kappa_mean_reduction( const double sigma, const double mean )
            : sigma_(sigma), mean_(mean), sum_(0), newn_(0)
      {  }

      inline bool evaluate( const T x )
      {
         const double x_ = double(x);
         if(::fabs(x_ - mean_) <= sigma_)
         {
            sum_ +=  x_;
            ++newn_;
         }
         return true;
      }

      inline double result() const
      {
         return sum_ / double(newn_);
      }

      inline int newn() const
      {
         return newn_;
      }

      inline void copyResult(const kappa_mean_reduction<T>& other)
      {
         sum_  = other.sum_;
         newn_ = other.newn_;
      }

      enum { isVectorizable = 0 };
      typedef double value_type;

   private:
      const double sigma_;
      const double mean_;
      double sum_;
      int    newn_;
};

template<class T>
class kappa_mean_reduction_nan : public _reduction_base
{
   public:
      kappa_mean_reduction_nan( const double sigma, const double mean, const T nan )
            : sigma_(sigma), mean_(mean), nan_(nan), sum_(0), newn_(0)
      {  }

      inline bool evaluate( const T x )
      {
         if(x != nan_)
         {
            const double x_ = double(x);
            if(::fabs(x_ - mean_) <= sigma_)
            {
               sum_ += x_;
               ++newn_;
            }
         }
         return true;
      }

      inline double result() const
      {
         if(newn_ > 0)
            return sum_ / double(newn_);
         return double(nan_);
      }

      inline int newn() const
      {
         return newn_;
      }

      inline void copyResult(const kappa_mean_reduction_nan<T>& other)
      {
         sum_  = other.sum_;
         newn_ = other.newn_;
      }

      enum { isVectorizable = 0 };
      typedef double value_type;

   private:
      const double sigma_;
      const double mean_;
      const T nan_;
      double sum_;
      int    newn_;
};

template<class T>
class kappa_sigma_reduction : public _reduction_base
{
   public:
      kappa_sigma_reduction( const double sigma, const double mean )
            : sigma_(sigma), mean_(mean), sum_(0.0), newn_(0)
      {  }

      inline bool evaluate( const T x )
      {
         const double tmp = ::fabs(double(x) - mean_); // difference
         if(tmp <= sigma_) // within sigma ?
         {
            sum_ +=  tmp * tmp; // add square to sum
            ++newn_; // increment counter
         }
         return true;
      }

      inline double result() const
      {
         if(newn_ > 1)
            return sum_ / double(newn_-1);
         return 0.0;
      }

      inline double sum() const
      {
         return sum_;
      }

      inline int newn() const
      {
         return newn_;
      }

      inline void copyResult(const kappa_sigma_reduction<T>& other)
      {
         sum_  = other.sum_;
         newn_ = other.newn_;
      }

      enum { isVectorizable = 0 };
      typedef double value_type;

   private:
      const double sigma_;
      const double mean_;
      double sum_;
      int    newn_;
};

template<class T>
class kappa_sigma_reduction_nan : public _reduction_base
{
   public:
      kappa_sigma_reduction_nan( const double sigma, const double mean, const T nan)
            : sigma_(sigma), mean_(mean), nan_(nan), sum_(0.0), newn_(0)
      {  }

      inline bool evaluate( const T x )
      {
         if(x != nan_)
         {
            const double tmp = ::fabs(double(x) - mean_); // difference
            if(tmp <= sigma_)
            {
               sum_ += tmp * tmp; // add to sum
               ++newn_; // increment counter
            }
         }
         return true;
      }

      inline double result() const
      {
         if(newn_ > 1)
            return sum_ / double(newn_ - 1);
         if(newn_ == 1)
            return 0.0;
         return double(nan_);
      }

      inline double sum() const
      {
         return sum_;
      }

      inline int newn() const
      {
         return newn_;
      }

      inline void copyResult(const kappa_sigma_reduction_nan<T>& other)
      {
         sum_  = other.sum_;
         newn_ = other.newn_;
      }

      enum { isVectorizable = 0 };
      typedef double value_type;

   private:
      const double sigma_;
      const double mean_;
      const T nan_;
      double sum_;
      int    newn_;
};


//---------------------------------------------------------------------
// HISTOGRAM
// min incl., max excl., range = [min, max)
// one bin = [ (max - min) / nobins )
//---------------------------------------------------------------------


template<class T>
class histogram_reduction : public _reduction_base
{
   public:
      histogram_reduction(const int no_bins, const T min, const T max) :
            no_bins_(no_bins), min_(min), max_(max),
            range_( max_ - min_ ),
            step_( double(range_) / double(no_bins_) ), sum_(0), histogram_( no_bins_ )
      {
         histogram_ = 0;
      }

      inline bool evaluate( const T x )
      {
         if( (x >= min_) && (x < max_) )
         {
            const int hist_offset = int( double( x - min_ ) / step_ );
            ++(histogram_( hist_offset + 1));
            ++sum_;
         }
         return true;
      }

      inline MArray<int,1>result() const
      {
         return histogram_;
      }

      inline T min() const
      {
         return min_;
      }

      inline T max() const
      {
         return max_;
      }

      inline int sum() const
      {
         return sum_;
      }

      inline double step() const
      {
         return step_;
      }

      inline void copyResult(const histogram_reduction<T>& other)
      {
         no_bins_ = other.no_bins_;
         step_    = other.step_;
         sum_     = other.sum_;
         histogram_.makeReference(other.histogram_);
      }

      enum { isVectorizable = 0 };
      typedef int value_type;

   private:
      int no_bins_;
      const T min_;
      const T max_;
      const T range_;
      double step_;
      int sum_;
      MArray<int, 1>histogram_;
};

template<class T>
class histogram_reduction_nan : public _reduction_base
{
   public:
      histogram_reduction_nan(const int no_bins, const T min, const T max, const T nan) :
            no_bins_(no_bins), min_(min), max_(max), nan_(nan),
            range_( max_ - min_ ),
            step_( double(range_) / double(no_bins_) ), sum_(0), histogram_( no_bins_ )
      {
         histogram_ = 0;
      }

      inline bool evaluate( const T x )
      {
         if( (x != nan_) && (x >= min_) && (x < max_) )
         {
            const int hist_offset = int( double( x - min_ ) / step_ );
            ++(histogram_( hist_offset + 1));
            ++sum_;
         }
         return true;
      }

      inline MArray<int,1>result() const
      {
         return histogram_;
      }

      inline T min() const
      {
         return min_;
      }

      inline T max() const
      {
         return max_;
      }

      inline int sum() const
      {
         return sum_;
      }

      inline double step() const
      {
         return step_;
      }

      inline void copyResult(const histogram_reduction_nan<T>& other)
      {
         no_bins_ = other.no_bins_;
         step_    = other.step_;
         sum_     = other.sum_;
         histogram_.makeReference(other.histogram_);
      }

      enum { isVectorizable = 0 };
      typedef int value_type;

   private:
      int no_bins_;
      const T min_;
      const T max_;
      const T nan_;
      const T range_;
      double step_;
      int sum_;
      MArray<int, 1>histogram_;
};


//---------------------------------------------------------------------
// KAPPA-SIGMA CLIPPED MEDIAN
//---------------------------------------------------------------------

// assuming a sorted 1dim array
template<class T>
class kappa_median_reduction : public _reduction_base
{
   public:
      kappa_median_reduction(const MArray<T, 1>& a,
                             const double min, const double max)
            : array_(a), min_(T(min)), max_(T(max)),
            n_(0), lower_boundary_(0), upper_boundary_(0)
      { }

      inline bool evaluate( const T x )
      {
         ++n_;
         if( !lower_boundary_ )
         {
            if( x >= min_ )
               lower_boundary_ = n_;
         }
         else
            if( x > max_ )
            {
               upper_boundary_ = n_ - 1;
               return false;
            }
         return true;
      }

      inline double result()
      {
         if(!lower_boundary_)
            lower_boundary_ = 1;
         if(!upper_boundary_)
            upper_boundary_ = array_.nelements();
         const int range = upper_boundary_ - lower_boundary_ + 1;
         const bool is_odd = range % 2;
         const int med_offset = lower_boundary_ + range / 2;
         if(is_odd)
            return double(array_(med_offset));
         return (double(array_(med_offset)) + double(array_(med_offset-1))) / 2.0;
      }

      inline void copyResult(const kappa_median_reduction<T>& other)
      {
         n_              = other.n_;
         lower_boundary_ = other.lower_boundary_;
         upper_boundary_ = other.upper_boundary_;
      }

      enum { isVectorizable = 0 };
      typedef double value_type;

   private:
      const MArray<T, 1> array_;
      const T min_;
      const T max_;
      int n_;
      int lower_boundary_;
      int upper_boundary_;
};


//---------------------------------------------------------------------
// MINIMUM VALUE
//---------------------------------------------------------------------

template<class T>
class min_reduction : public _reduction_base
{
   public:
      min_reduction()
            : min_(numeric_limits<T>::max())
      {  }

      inline bool evaluate( const T x )
      {
         if( x < min_ )
            min_ = x;
         return true;
      }

      inline T result() const
      {
         return min_;
      }

      inline void copyResult(const min_reduction<T>& other)
      {
         min_ = other.min_;
      }

      typedef T value_type;
#ifdef LTL_USE_SIMD
      typedef T vec_calc_type;
      enum { isVectorizable = min_reduction_vec<T>::isVectorizable };
      typedef min_reduction_vec<T> vec_reduction;
      inline void copyResult( const vec_reduction& other )
      {
         min_ = other.result();
      }
#endif
   private:
      T min_;
};

template<class T>
class min_reduction_nan : public _reduction_base
{
   public:
      min_reduction_nan(const T nan)
            : nan_(nan), min_(numeric_limits<T>::max())
      {  }

      inline bool evaluate( const T x )
      {
         if( (x != nan_) && (x < min_) )
            min_ = x;
         return true;
      }

      inline T result() const
      {
         return min_;
      }

      inline void copyResult(const min_reduction_nan<T>& other)
      {
         min_ = other.min_;
      }

      typedef T value_type;
   private:
      const T nan_;
      T min_;
};


//---------------------------------------------------------------------
// MAXIMUM VALUE
//---------------------------------------------------------------------

template<class T>
class max_reduction : public _reduction_base
{
   public:
      max_reduction()
            : max_(-numeric_limits<T>::max())
      {  }

      inline bool evaluate( const T x )
      {
         if( x > max_ )
            max_ = x;
         return true;
      }

      inline T result() const
      {
         return max_;
      }

      inline void copyResult(const max_reduction<T>& other)
      {
         max_ = other.max_;
      }

      typedef T value_type;
#ifdef LTL_USE_SIMD
      typedef T vec_calc_type;
      enum { isVectorizable = max_reduction_vec<T>::isVectorizable };
      typedef max_reduction_vec<T> vec_reduction;
      inline void copyResult( const vec_reduction& other )
      {
         max_ = other.result();
      }
#endif
   private:
      T max_;
};

template<class T>
class max_reduction_nan : public _reduction_base
{
   public:
      max_reduction_nan(const T nan)
            : nan_(nan), max_(-numeric_limits<T>::max())
      {  }

      inline bool evaluate( const T x )
      {
         if( (x != nan_) && (x > max_) )
            max_ = x;
         return true;
      }

      inline T result() const
      {
         return max_;
      }

      inline void copyResult(const max_reduction_nan<T>& other)
      {
         max_ = other.max_;
      }

      typedef T value_type;
   private:
      const T nan_;
      T max_;
};


//---------------------------------------------------------------------
// COUNT LOGICAL TRUE
//---------------------------------------------------------------------

template<class T>
class count_reduction : public _reduction_base
{
   public:
      count_reduction()
            : n_(0)
      {  }

      inline bool evaluate( const T x )
      {
         if(x != T(0))
            ++n_;
         return true;
      }

      inline int result() const
      {
         return n_;
      }

      inline void copyResult(const count_reduction<T>& other)
      {
         n_ = other.n_;
      }

      typedef int value_type;
#ifdef LTL_USE_SIMD
      typedef T vec_calc_type;
      enum { isVectorizable = count_reduction_vec<T>::isVectorizable };
      typedef count_reduction_vec<T> vec_reduction;
      inline void copyResult( const vec_reduction& other )
      {
         n_ = other.result();
      }
#endif
   private:
      int n_;
};


//---------------------------------------------------------------------
// ALL TRUE
//---------------------------------------------------------------------

template<class T>
class allof_reduction : public _reduction_base
{
   public:
      allof_reduction()
            : tmp_(true)
      {  }

      inline bool evaluate( const T x )
      {
         if( !x )
            tmp_ = false;
         return tmp_;
      }

      inline bool result() const
      {
         return tmp_;
      }

      inline void copyResult(const allof_reduction<T>& other)
      {
         tmp_ = other.tmp_;
      }

      typedef bool value_type;
#ifdef LTL_USE_SIMD
      typedef T vec_calc_type;
      enum { isVectorizable = allof_reduction_vec<T>::isVectorizable };
      typedef allof_reduction_vec<T> vec_reduction;
      inline void copyResult( const vec_reduction& other )
      {
         tmp_ = other.result();
      }
#endif
   private:
      bool tmp_;
};


//---------------------------------------------------------------------
// NONE TRUE
//---------------------------------------------------------------------

template<class T>
class noneof_reduction : public _reduction_base
{
   public:
      noneof_reduction()
            : tmp_(true)
      {  }

      inline bool evaluate( const T x )
      {
         if( x )
            tmp_ = false;
         return tmp_;
      }

      inline bool result() const
      {
         return tmp_;
      }

      inline void copyResult(const noneof_reduction<T>& other)
      {
         tmp_ = other.tmp_;
      }

      typedef bool value_type;
#ifdef LTL_USE_SIMD
      typedef T vec_calc_type;
      enum { isVectorizable = noneof_reduction_vec<T>::isVectorizable };
      typedef noneof_reduction_vec<T> vec_reduction;
      inline void copyResult( const vec_reduction& other )
      {
         tmp_ = other.result();
      }
#endif
   private:
      bool tmp_;
};


//---------------------------------------------------------------------
// AT LEAST 1 TRUE
//---------------------------------------------------------------------

template<class T>
class anyof_reduction : public _reduction_base
{
   public:
      anyof_reduction()
            : tmp_(false)
      {  }

      inline bool evaluate( const T x )
      {
         if( x )
            tmp_ = true;
         return !tmp_;
      }

      inline bool result() const
      {
         return tmp_;
      }

      inline void copyResult(const anyof_reduction<T>& other)
      {
         tmp_ = other.tmp_;
      }

      typedef bool value_type;
#ifdef LTL_USE_SIMD
      typedef T vec_calc_type;
      enum { isVectorizable = anyof_reduction_vec<T>::isVectorizable };
      typedef anyof_reduction_vec<T> vec_reduction;
      inline void copyResult( const vec_reduction& other )
      {
         tmp_ = other.result();
      }
#endif
   private:
      bool tmp_;
};

#if defined(LTL_USE_SIMD) && defined(__SSE2__)
#include <ltl/marray/reductions_sse.h>
#endif

}

#endif

