/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: mrqhesse.h 476 2010-11-12 06:00:58Z drory $
 * ---------------------------------------------------------------------
 *
 * Copyright (C)  Claus A. Goessl <cag@usm.uni-muenchen.de>
 *                         Niv Drory <drory@mpe.mpg.de>
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

#ifndef __LTL_MRQHESSE__
#define __LTL_MRQHESSE__

template<class TPAR, int NPAR>
class tMatHesse;

template<class TPAR, int NPAR, bool unroll>
class tMatHesseSplit;

template<class TPAR, int NPAR, int L, int K>
class tNMatHesse;

template<class TPAR, int NPAR>
class tMatFillHesse;

template<class TPAR, int NPAR, bool unroll>
class tMatFillHesseSplit;

template<class TPAR, int NPAR, int L, int K>
class tNMatFillHesse;

template<class TPAR, int NPAR>
class tMatClearHesse;

template<class TPAR, int NPAR, bool unroll>
class tMatClearHesseSplit;

template<class TPAR, int NPAR, int L, int K>
class tNMatClearHesse;

template<class TPAR, int NPAR>
class tMatHesse
{
   public:
      enum { static_size = (((NPAR + 5) * NPAR) / 2)};

      static inline void eval(const FVector<TPAR, NPAR>& restrict_ derivative, 
                              const TPAR df, const TPAR sig2,
                              FMatrix<TPAR, NPAR, NPAR>& restrict_ a,
                              FVector<TPAR, NPAR>& restrict_ b)
      {
         tMatHesseSplit<TPAR, NPAR,
                        (static_size <= LTL_TEMPLATE_LOOP_LIMIT)
                       >::eval(derivative, df, sig2, a, b);
      }
};

template<class TPAR, int NPAR, bool unroll>
class tMatHesseSplit
{ };

template<class TPAR, int NPAR>
class tMatHesseSplit<TPAR, NPAR, true>
{
   public:
      static inline void eval(const FVector<TPAR, NPAR>& restrict_ derivative,
                              const TPAR df, const TPAR sig2,
                              FMatrix<TPAR, NPAR, NPAR>& restrict_ a,
                              FVector<TPAR, NPAR>& restrict_ b)
      {
         tNMatHesse<TPAR, NPAR, NPAR, 0>::eval(derivative, df, sig2, a, b);
      }

};

template<class TPAR, int NPAR>
class tMatHesseSplit<TPAR, NPAR, false>
{
   public:
      static inline void eval(const FVector<TPAR, NPAR>& restrict_ derivative,
                              const TPAR df, const TPAR sig2,
                              FMatrix<TPAR, NPAR, NPAR>& restrict_ a,
                              FVector<TPAR, NPAR>& restrict_ b)
      {
         for(int l=1; l <= NPAR; ++l)
         {
            const TPAR wt = derivative(l) / sig2;
            for(int k=1; k <= l; ++k)
               a(l, k) += wt * derivative(k);
            b(l) += df * wt;
         }
      }
};


template<class TPAR, int NPAR, int L, int K>
class tNMatHesse
{
   public:
      static inline void eval(const FVector<TPAR, NPAR>& restrict_ derivative,
                              const TPAR wt,
                              FMatrix<TPAR, NPAR, NPAR>& restrict_ a)
      {
         tNMatHesse<TPAR, NPAR, L, K-1>::eval(derivative, wt, a);
         a(L, K) += wt * derivative(K);
      }
};

template<class TPAR, int NPAR, int L>
class tNMatHesse<TPAR, NPAR, L, 1>
{
   public:
      static inline void eval(const FVector<TPAR, NPAR>& restrict_ derivative,
                              const TPAR wt,
                              FMatrix<TPAR, NPAR, NPAR>& restrict_ a)
      {
         a(L, 1) += wt * derivative(1);
      }
};

template<class TPAR, int NPAR, int L>
class tNMatHesse<TPAR, NPAR, L, 0>
{
   public:
      static inline void eval(const FVector<TPAR, NPAR>& restrict_ derivative,
                              const TPAR df, const TPAR sig2,
                              FMatrix<TPAR, NPAR, NPAR>& restrict_ a,
                              FVector<TPAR, NPAR>& restrict_ b)
      {
         const TPAR wt = derivative(L) / sig2;
         tNMatHesse<TPAR, NPAR, L, L>::eval(derivative, wt, a);
         b(L) += df * wt;
         tNMatHesse<TPAR, NPAR, L-1, 0>::eval(derivative, df, sig2, a, b);
      }
};

template<class TPAR, int NPAR>
class tNMatHesse<TPAR, NPAR, 0, 0>
{
   public:
      static inline void eval(const FVector<TPAR, NPAR>& restrict_ derivative,
                              const TPAR df, const TPAR sig2,
                              FMatrix<TPAR, NPAR, NPAR>& restrict_ a,
                              FVector<TPAR, NPAR>& restrict_ b)
      { }
};

template<class TPAR, int NPAR>
class tMatFillHesse
{
   public:
      enum { static_size = (((NPAR - 1) * NPAR) / 2)};
      static inline void eval(FMatrix<TPAR, NPAR, NPAR>& restrict_ a)
      {
         tMatFillHesseSplit<TPAR, NPAR,
                            (static_size <= LTL_TEMPLATE_LOOP_LIMIT)
                           >::eval(a);
      }
};

template<class TPAR, int NPAR, bool unroll>
class tMatFillHesseSplit
{ };


template<class TPAR, int NPAR>
class tMatFillHesseSplit<TPAR, NPAR, true>
{
   public:
      static inline void eval(FMatrix<TPAR, NPAR, NPAR>& restrict_ a)
      {
         tNMatFillHesse<TPAR, NPAR, NPAR, 0>::eval(a);
      }
};

template<class TPAR, int NPAR>
class tMatFillHesseSplit<TPAR, NPAR, false>
{
   public:
      static inline void eval(FMatrix<TPAR, NPAR, NPAR>& restrict_ a)
      {
         for(int l=2; l <= NPAR; ++l)
            for(int k=1; k < l; ++k)
               a(k, l) = a(l, k);
      }
};

template<class TPAR, int NPAR, int L, int K>
class tNMatFillHesse
{
   public:
      static inline void eval(FMatrix<TPAR, NPAR, NPAR>& restrict_ a)
      {
         tNMatFillHesse<TPAR, NPAR, L, K-1>::eval(a);
         a(K, L) = a(L, K);
      }
};

template<class TPAR, int NPAR, int L>
class tNMatFillHesse<TPAR, NPAR, L, 1>
{
   public:
      static inline void eval(FMatrix<TPAR, NPAR, NPAR>& restrict_ a)
      {
         a(1, L) = a(L, 1);
      }
};

template<class TPAR, int NPAR, int L>
class tNMatFillHesse<TPAR, NPAR, L, 0>
{
   public:
      static inline void eval(FMatrix<TPAR, NPAR, NPAR>& a)
      {
         tNMatFillHesse<TPAR, NPAR, L, L-1>::eval(a);
         tNMatFillHesse<TPAR, NPAR, L-1, 0>::eval(a);
      }
};

template<class TPAR, int NPAR>
class tNMatFillHesse<TPAR, NPAR, 1, 0>
{
   public:
      static inline void eval(FMatrix<TPAR, NPAR, NPAR>& a)
      { }
};

template<class TPAR, int NPAR>
class tMatClearHesse
{
   public:
      enum { static_size = (((NPAR + 1) * NPAR) / 2)};
      static inline void eval(FMatrix<TPAR, NPAR, NPAR>& a)
      {
         tMatClearHesseSplit<TPAR, NPAR,
                             (static_size <= LTL_TEMPLATE_LOOP_LIMIT)
                            >::eval(a);
      }
};

template<class TPAR, int NPAR, bool unroll>
class tMatClearHesseSplit
{ };


template<class TPAR, int NPAR>
class tMatClearHesseSplit<TPAR, NPAR, true>
{
   public:
      static inline void eval(FMatrix<TPAR, NPAR, NPAR>& a)
      {
         tNMatClearHesse<TPAR, NPAR, NPAR, 0>::eval(a);
      }
};

template<class TPAR, int NPAR>
class tMatClearHesseSplit<TPAR, NPAR, false>
{
   public:
      static inline void eval(FMatrix<TPAR, NPAR, NPAR>& a)
      {
         for(int l=1; l <= NPAR; ++l)
            for(int k=1; k <= l; ++k)
               a(l, k) = TPAR(0);
      }
};

template<class TPAR, int NPAR, int L, int K>
class tNMatClearHesse
{
   public:
      static inline void eval(FMatrix<TPAR, NPAR, NPAR>& a)
      {
         tNMatClearHesse<TPAR, NPAR, L, K-1>::eval(a);
         a(L, K) = TPAR(0);
      }
};

template<class TPAR, int NPAR, int L>
class tNMatClearHesse<TPAR, NPAR, L, 1>
{
   public:
      static inline void eval(FMatrix<TPAR, NPAR, NPAR>& a)
      {
         a(L, 1) = TPAR(0);
      }
};

template<class TPAR, int NPAR, int L>
class tNMatClearHesse<TPAR, NPAR, L, 0>
{
   public:
      static inline void eval(FMatrix<TPAR, NPAR, NPAR>& a)
      {
         tNMatClearHesse<TPAR, NPAR, L, L>::eval(a);
         tNMatClearHesse<TPAR, NPAR, L-1, 0>::eval(a);
      }
};

template<class TPAR, int NPAR>
class tNMatClearHesse<TPAR, NPAR, 0, 0>
{
   public:
      static inline void eval(FMatrix<TPAR, NPAR, NPAR>& a)
      { }
};

#endif   // __LTL_MRQHESSE__

