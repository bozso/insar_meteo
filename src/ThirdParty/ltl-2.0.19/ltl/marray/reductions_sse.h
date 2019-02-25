/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: reductions_sse.h 368 2008-07-17 14:52:13Z drory $
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


#ifndef __LTL_REDUCTIONS_SSE__
#define __LTL_REDUCTIONS_SSE__


/*! \file reductions_sse.h
    SSE2 or higher vectorized versions of reduction operations
    defined in \file reductions.h.

    These get used instead of the non-vectorized ones if
    the expression argument is vectorizeable and if the reduction
    operation from reductions.h defines a vectorized version.

    We need both vector and scalar versions of \code evaluate()\endcode
    because we need to handle the case that the length of the
    expression is not a multiple of the vector size. The scalar version
    will be called to handle the border elements.

    When \code result() \endcode is called, we combine the scalar and
    vector partial result to a final result which we return.

    The code that decides what operation to use is in eval_reduc.h
 */

//---------------------------------------------------------------------
// SUM
//---------------------------------------------------------------------

template<class T>
class sum_reduction_vec : public _reduction_base
{
   public:
      enum { isVectorizable = 1 };
      sum_reduction_vec()
            : vsum_(T_VEC_ZERO(T)), sum_(0)
      {  }

      sum_reduction_vec( const sum_reduction<T>& s )
            : vsum_(T_VEC_ZERO(T)), sum_(0)
      {  }

      inline bool evaluate( const T_VEC_TYPE(T) x )
      {
         vsum_ = __ltl_TAdd<T,T>::eval_vec( vsum_, x );
         return true;
      }

      inline bool evaluate( const T x )
      {
         sum_ += x;
         return true;
      }

      inline T result() const
      {
         T_VEC_UNION(T) u;
         u.v = vsum_;
         T s = 0;
         for( int i=0; i<T_VEC_LEN(T); ++i )
            s += u.a[i];
         return s + sum_;
      }

   private:
      T_VEC_TYPE(T) vsum_;
      T sum_;
};

template<class T>
class sum_reduction_nan_vec : public _reduction_base
{
   public:
      enum { isVectorizable = 1 };
      sum_reduction_nan_vec()
            : vsum_(T_VEC_ZERO(T)), vnan_(T_VEC_ZERO(T)), vn_(T_VEC_ZERO(T_INT_TYPE(T))),
            sum_(0), nan_(0), n_(0)
      {  }

      sum_reduction_nan_vec( const sum_reduction_nan<T>& s )
            : vsum_(T_VEC_ZERO(T)), vnan_(VEC_INIT(T,s.nan_)), vn_(T_VEC_ZERO(T_INT_TYPE(T))),
            sum_(0), nan_(s.nan_), n_(0)
      {  }

      inline bool evaluate( const T_VEC_TYPE(T) x )
      {
         const T_VEC_TYPE(T) m = __ltl_TEQ<T,T>::eval_vec( vnan_, x );
         // vec_sel(a,b,m) reduces to this for b=0.
         const T_VEC_TYPE(T) s = sse_vec_andnot(m,x);
         vsum_ = __ltl_TAdd<T,T>::eval_vec( vsum_, s );

         // count up the number of data != nan
         // note that the mask is 0xFFFF = -1 so we count negative
         vn_ = __ltl_TSub<T_INT_TYPE(T),T_INT_TYPE(T)>::eval_vec( vn_, (T_INT_VEC_TYPE(T))sse_vec_not(m) );
         return true;
      }

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
         T_VEC_UNION(T) u;
         u.v = vsum_;
         T s = 0;
         for( int i=0; i<T_VEC_LEN(T); ++i )
            s += u.a[i];
         return s + sum_;
      }

      inline int n() const
      {
         T_VEC_UNION(T_INT_TYPE(T)) u;
         u.v = vn_;
         int s = 0;
         for( int i=0; i<T_VEC_LEN(T); ++i )
            s += u.a[i];
         return s + n_;
      }

   private:
      T_VEC_TYPE(T) vsum_;
      const T_VEC_TYPE(T) vnan_;
      T_INT_VEC_TYPE(T) vn_;
      T sum_;
      const T nan_;
      int n_;
};


//---------------------------------------------------------------------
// AVERAGE
//---------------------------------------------------------------------

template<class T>
class avg_reduction_vec : public _reduction_base
{
   public:
      enum { isVectorizable = 1 };
      avg_reduction_vec()
            : vsum_(T_VEC_ZERO(T)), sum_(0), n_(0)
      {  }

      avg_reduction_vec( const avg_reduction<T>& s )
            : vsum_(T_VEC_ZERO(T)), sum_(0), n_( s.n_ )
      {  }

      inline bool evaluate( const T_VEC_TYPE(T) x )
      {
         vsum_ = __ltl_TAdd<T,T>::eval_vec( vsum_, x );
         return true;
      }

      inline bool evaluate( const T x )
      {
         sum_ += x;
         return true;
      }

      inline T result() const
      {
         return sum()/n_;
      }

      inline int n() const
      {
         return n_;
      }

      inline T sum() const
      {
         T_VEC_UNION(T) u;
         u.v = vsum_;
         T s = 0;
         for( int i=0; i<T_VEC_LEN(T); ++i )
            s += u.a[i];
         return s + sum_;
      }

   private:
      T_VEC_TYPE(T) vsum_;
      T sum_;
      int n_;
};

template<class T>
class avg_reduction_nan_vec : public _reduction_base
{
   public:
      enum { isVectorizable = 1 };
      avg_reduction_nan_vec()
            : vsum_(T_VEC_ZERO(T)), vnan_(T_VEC_ZERO(T)), vn_(T_VEC_ZERO(T_INT_TYPE(T))),
            sum_(0), nan_(0), n_(0)
      {  }

      avg_reduction_nan_vec( const avg_reduction_nan<T>& s )
            : vsum_(T_VEC_ZERO(T)), vnan_(VEC_INIT(T,s.nan_)), vn_(T_VEC_ZERO(T_INT_TYPE(T))),
            sum_(0), nan_(s.nan_), n_(0)
      {  }

      inline bool evaluate( const T_VEC_TYPE(T) x )
      {
         const T_VEC_TYPE(T) m = __ltl_TEQ<T,T>::eval_vec( vnan_, x );
         // vec_sel(a,b,m) reduces to this for b=0.
         const T_VEC_TYPE(T) s = sse_vec_andnot(m,x);
         vsum_ = __ltl_TAdd<T,T>::eval_vec( vsum_, s );

         // count up the number of data != nan
         // note that the mask is 0xFFFF = -1 so we count negative
         vn_ = __ltl_TSub<T_INT_TYPE(T),T_INT_TYPE(T)>::eval_vec( vn_, (T_INT_VEC_TYPE(T))sse_vec_not(m) );
         return true;
      }

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
         return sum()/(T)n();
      }

      inline T sum() const
      {
         T_VEC_UNION(T) u;
         u.v = vsum_;
         T s = 0;
         for( int i=0; i<T_VEC_LEN(T); ++i )
            s += u.a[i];
         return s + sum_;
      }

      inline int n() const
      {
         T_VEC_UNION(T_INT_TYPE(T)) u;
         u.v = vn_;
         int s = 0;
         for( int i=0; i<T_VEC_LEN(T); ++i )
            s += u.a[i];
         return s + n_;
      }

   private:
      T_VEC_TYPE(T) vsum_;
      const T_VEC_TYPE(T) vnan_;
      T_INT_VEC_TYPE(T) vn_;
      T sum_;
      const T nan_;
      int n_;
};


//---------------------------------------------------------------------
// VARIANCE (RMS**2)
//---------------------------------------------------------------------

template<class T>
class variance_reduction_vec : public _reduction_base
{
   public:
      enum { isVectorizable = 1 };
      variance_reduction_vec( const int n, const double avg )
            : sum_(0.0), n_(n), avg_(avg),
            vsum_(T_VEC_ZERO(T))
      {  }

      variance_reduction_vec( const variance_reduction<T>& s )
            : sum_(0.0), n_(s.n_), avg_(s.avg_),
            vsum_(T_VEC_ZERO(T))
      {  }

      inline bool evaluate( const T x )
      {
         const double tmp = double(x) - avg_;
         sum_ += tmp * tmp;
         return true;
      }

      inline bool evaluate( const T_VEC_TYPE(T) x )
      {
         T_VEC_TYPE(T) tmp = __ltl_TSub<T,T>::eval_vec( x, VEC_INIT(T,avg_) );
         vsum_ = __ltl_TAdd<T,T>::eval_vec( vsum_, __ltl_TMul<T,T>::eval_vec(tmp,tmp) );
         return true;
      }

      inline T result() const
      {
         if( n_ > 1 )
            return sum()/(n_-1);
         else
            return 0;
      }

      inline int n() const
      {
         return n_;
      }

      inline T sum() const
      {
         T_VEC_UNION(T) u;
         u.v = vsum_;
         T s = 0;
         for( int i=0; i<T_VEC_LEN(T); ++i )
            s += u.a[i];
         return s + sum_;
      }

   private:
      T sum_;
      const int n_;
      const T avg_;
      T_VEC_TYPE(T) vsum_;
};

template<class T>
class variance_reduction_nan_vec : public _reduction_base
{
   public:
      variance_reduction_nan_vec( const double avg, const T nan )
            : vsum_(T_VEC_ZERO(T)), vnan_(VEC_INIT(T,nan_)), vn_(T_VEC_ZERO(T_INT_TYPE(T))),
            sum_(0.0), n_(0), avg_(avg), nan_(nan)
      {  }

      variance_reduction_nan_vec( const variance_reduction_nan<T>& s )
            : vsum_(T_VEC_ZERO(T)), vnan_(VEC_INIT(T,s.nan_)), vn_(T_VEC_ZERO(T_INT_TYPE(T))),
            sum_(0.0), n_(0), avg_(s.avg_), nan_(s.nan_)
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

      inline bool evaluate( const T_VEC_TYPE(T) x )
      {
         const T_VEC_TYPE(T) m = __ltl_TEQ<T,T>::eval_vec( vnan_, x );
         T_VEC_TYPE(T) tmp = __ltl_TSub<T,T>::eval_vec( x, VEC_INIT(T,avg_) );
         // vec_sel(a,b,m) reduces to this for b=0.
         const T_VEC_TYPE(T) s = sse_vec_andnot(m,__ltl_TMul<T,T>::eval_vec(tmp,tmp));
         vsum_ = __ltl_TAdd<T,T>::eval_vec( vsum_, s );

         // count up the number of data != nan
         // note that the mask is 0xFFFF = -1 so we count negative
         vn_ = __ltl_TSub<T_INT_TYPE(T),T_INT_TYPE(T)>::eval_vec( vn_, (T_INT_VEC_TYPE(T))sse_vec_not(m) );
         return true;
      }

      inline T result() const
      {
         if( n_ > 1 )
            return sum()/(n_-1);
         if(n_ == 1)
            return 0.0;
         return nan_;
      }

      inline T sum() const
      {
         T_VEC_UNION(T) u;
         u.v = vsum_;
         T s = 0;
         for( int i=0; i<T_VEC_LEN(T); ++i )
            s += u.a[i];
         return s + sum_;
      }

      inline int n() const
      {
         T_VEC_UNION(T_INT_TYPE(T)) u;
         u.v = vn_;
         int s = 0;
         for( int i=0; i<T_VEC_LEN(T); ++i )
            s += u.a[i];
         return s + n_;
      }

   private:
      T_VEC_TYPE(T) vsum_;
      T_VEC_TYPE(T) vnan_;
      T_INT_VEC_TYPE(T) vn_;
      T sum_;
      int n_;
      const T avg_;
      const T nan_;
};


//---------------------------------------------------------------------
// MINIMUM VALUE
//---------------------------------------------------------------------

template<typename T>
class min_reduction_vec : public _reduction_base
{
   public:
      enum { isVectorizable = 1 };
      min_reduction_vec()
            : v_(VEC_INIT(T,numeric_limits<T>::max())),
            m_(numeric_limits<T>::max())
      {  }

      min_reduction_vec( const min_reduction<T>& s )
            : v_(VEC_INIT(T,numeric_limits<T>::max())),
            m_(numeric_limits<T>::max())
      {  }

      inline bool evaluate( const T_VEC_TYPE(T) x )
      {
         v_ = sse_vec_min( x, v_ );
         return true;
      }

      inline bool evaluate( const T x )
      {
         m_ = std::min( x, m_ );
         return true;
      }

      inline T result() const
      {
         T_VEC_UNION(T) u;
         u.v = v_;
         T s = u.a[0];
         for( int i=1; i<T_VEC_LEN(T); ++i )
            if( u.a[i] < s )
               s = u.a[i];
         return std::min( s, m_ );
      }

   private:
      T_VEC_TYPE(T) v_;
      T m_;
};

//---------------------------------------------------------------------
// MAXIMUM VALUE
//---------------------------------------------------------------------

template<typename T>
class max_reduction_vec : public _reduction_base
{
   public:
      enum { isVectorizable = 1 };
      max_reduction_vec()
            : v_(VEC_INIT(T,numeric_limits<T>::min())),
            m_(numeric_limits<T>::min())
      {  }

      max_reduction_vec( const max_reduction<T>& m )
            : v_(VEC_INIT(T,numeric_limits<T>::min())),
            m_(numeric_limits<T>::min())
      {  }

      inline bool evaluate( const T_VEC_TYPE(T) x )
      {
         v_ = sse_vec_max( x, v_ );
         return true;
      }

      inline bool evaluate( const T x )
      {
         m_ = std::max( x, m_ );
         return true;
      }

      inline T result() const
      {
         T_VEC_UNION(T) u;
         u.v = v_;
         T s = u.a[0];
         for( int i=1; i<T_VEC_LEN(T); ++i )
            if( u.a[i] > s )
               s = u.a[i];
         return std::max( s, m_ );
      }

   private:
      T_VEC_TYPE(T) v_;
      T m_;
};


//---------------------------------------------------------------------
// COUNT LOGICAL TRUE
//---------------------------------------------------------------------

template<class T>
class count_reduction_vec : public _reduction_base
{
   public:
      enum { isVectorizable = 1 };
      count_reduction_vec()
            : vn_(T_VEC_ZERO(T_INT_TYPE(T))), n_(0)
      {  }

      count_reduction_vec( const count_reduction<T>& s )
            : vn_(T_VEC_ZERO(T_INT_TYPE(T))), n_(0)
      {  }

      inline bool evaluate( const T x )
      {
         if(x != T(0))
            ++n_;
         return true;
      }

      inline bool evaluate( const T_VEC_TYPE(T) x )
      {
         const T_INT_VEC_TYPE(T) m = (T_INT_VEC_TYPE(T))__ltl_TNE<T,T>::eval_vec( x, T_VEC_ZERO(T) );
         // count up the number of data != 0
         // note that the mask is 0xFFFF = -1 so we count negative
         vn_ = __ltl_TSub<T_INT_TYPE(T),T_INT_TYPE(T)>::eval_vec( vn_, m );
         return true;
      }

      inline int result() const
      {
         T_VEC_UNION(T_INT_TYPE(T)) u;
         u.v = vn_;
         int s = 0;
         for( int i=0; i<T_VEC_LEN(T); ++i )
            s += u.a[i];
         return s + n_;
      }

   private:
      T_INT_VEC_TYPE(T) vn_;
      int n_;
};


//---------------------------------------------------------------------
// ALL TRUE
//---------------------------------------------------------------------

template<class T>
class allof_reduction_vec : public _reduction_base
{
   public:
      enum { isVectorizable = 1 };
      allof_reduction_vec()
            : tmp_(true)
      {  }

      allof_reduction_vec( const allof_reduction<T>& other )
            : tmp_(true)
      {  }

      inline bool evaluate( const T x )
      {
         if( !x )
            tmp_ = false;
         return tmp_;
      }

      inline bool evaluate( const T_VEC_TYPE(T) x )
      {
         const T_VEC_TYPE(T) m = __ltl_TEQ<T,T>::eval_vec( x, T_VEC_ZERO(T) );
         tmp_ = !sse_movemask( m );
         return tmp_;
      }

      inline bool result() const
      {
         return tmp_;
      }

   private:
      bool tmp_;
};



//---------------------------------------------------------------------
// NONE TRUE
//---------------------------------------------------------------------

template<class T>
class noneof_reduction_vec : public _reduction_base
{
   public:
      enum { isVectorizable = 1 };
      noneof_reduction_vec()
            : tmp_(true)
      {  }

      noneof_reduction_vec( const noneof_reduction<T>& other )
            : tmp_(true)
      {  }

      inline bool evaluate( const T x )
      {
         if( x )
            tmp_ = false;
         return tmp_;
      }

      inline bool evaluate( const T_VEC_TYPE(T) x )
      {
         const T_VEC_TYPE(T) m = __ltl_TNE<T,T>::eval_vec( x, T_VEC_ZERO(T) );
         tmp_ = !sse_movemask( m );
         return tmp_;
      }

      inline bool result() const
      {
         return tmp_;
      }

   private:
      bool tmp_;
};


//---------------------------------------------------------------------
// AT LEAST 1 TRUE
//---------------------------------------------------------------------

template<class T>
class anyof_reduction_vec : public _reduction_base
{
   public:
      enum { isVectorizable = 1 };
      anyof_reduction_vec()
            : tmp_(false)
      {  }

      anyof_reduction_vec( const anyof_reduction<T>& other )
            : tmp_(false)
      {  }

      inline bool evaluate( const T x )
      {
         if( x )
            tmp_ = true;
         return !tmp_;
      }

      inline bool evaluate( const T_VEC_TYPE(T) x )
      {
         const T_VEC_TYPE(T) m = __ltl_TNE<T,T>::eval_vec( x, T_VEC_ZERO(T) );
         tmp_ = sse_movemask( m );
         return !tmp_;
      }

      inline bool result() const
      {
         return tmp_;
      }

   private:
      bool tmp_;
};


#endif
