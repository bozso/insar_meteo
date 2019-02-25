/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testconvolve.cpp 551 2015-02-03 16:04:19Z drory $
* ---------------------------------------------------------------------
*
* Copyright (C) Niv Drory <drory@mpe.mpg.de>
*               Claus A. Goessl <cag@usm.uni-muenchen.de>
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

#define LTL_RANGE_CHECKING
//#define LTL_USE_SIMD
//#define LTL_DEBUG_EXPRESSIONS

#include <ltl/marray.h>
#include <ltl/marray_io.h>
#include <ltl/statistics.h>
#include <ltl/fvector.h>
#include <ltl/convolve.h>

#include <iostream>

using namespace ltl;

using std::cout;
using std::endl;

struct GaussianKernel1D
{
      GaussianKernel1D (const double s, const double k) :
            sigma_ (s), extent_ (k)
      {
         norm_ = 0.0;
         for (int i = -extent_; i<=extent_; ++i)
            norm_ += exp (-0.5*(double (i*i))/(sigma_*sigma_));
      }

      template<typename Iter>
      inline typename Iter::value_type eval (Iter& A) const
      {
         double r = 0.0;
         for (int i = -extent_; i<=extent_; ++i)
            r += exp (-0.5*(double (i*i))/(sigma_*sigma_))*OFFSET1(A, i);

         return typename Iter::value_type (r/norm_);
      }

      double sigma_, norm_;
      int extent_;
};
namespace ltl
{
template <typename Iter>
struct kernel_return_type<Iter,GaussianKernel1D>
{
   typedef typename Iter::value_type  value_type;
};
}

void test1d();
void test2d();
void test3d();
void test_with_index();
void test_grad();
void test_gaussian();

int main(int argc, char **argv)
{
   test1d();
   test2d();
   test3d();
   test_with_index();
   test_grad();
   test_gaussian();
}

void test1d()
{
   cerr << "Testing MArray convolution expressions in 1-D ..." << endl;

   MArray<float, 1> A(10);
   A = 0.0f;
   A(5) = 1.0f;

   MArray<float, 1> B;
   B = convolve(A, cderiv1(1));
   //cout << B << endl;
   LTL_ASSERT_( (B(4)==1 && B(6)==-1), "1-D MArray convolve failed.");
   B(4) = B(6) = 0.0f;
   LTL_ASSERT_( noneof(B), "1-D MArray convolve failed.");

   B = 0.0f;
   A = indexPosFlt(A,1);
   float n = 1.0f/(2.0f*1.0f);  // 1/(2h)
   B = n*convolve(A, cderiv1(1));
   LTL_ASSERT_( (B(1)==0 && B(10)==0), "1-D MArray convolve failed.");
   B(1) = B(10) = 1.0f;
   LTL_ASSERT_( allof(B==1.0f), "1-D MArray convolve failed.");
   //cout << B << endl;
}

void test2d()
{
   cerr << "Testing MArray convolution expressions in 2-D ..." << endl;

   MArray<float,2> A(10,10);
   A = 0.0f;
   A(5,5) = 1.0f;

   MArray<float,2> B;
   B = convolve(A, cderiv1(2));
   //cout << B << endl;
   LTL_ASSERT_( (B(5,4)==1 && B(5,6)==-1), "2-D MArray convolve with contiguous memory failed.");
   B(5,4) = B(5,6)=0;
   LTL_ASSERT_( noneof(B), "2-D MArray convolve with contiguous memory failed.");

   MArray<float,2> C(10,10);
   C = 0.0f;
   C(Range(3,8), Range::all()) = convolve(A(Range(3,8), Range::all()), cderiv1(2));
   //cout << C << endl;
   LTL_ASSERT_( (C(5,4)==1 && C(5,6)==-1), "2-D MArray convolve with non-contiguous memory failed.");
   C(5,4) = C(5,6) = 0;
   LTL_ASSERT_( noneof(C), "2-D MArray convolve with non-contiguous memory failed.");
}

void test3d()
{
   cerr << "Testing MArray convolution expressions in 3-D ..." << endl;

   MArray<float,3> A(10,10,6);
   A = 0.0f;
   A(5,5,3) = 1.0f;

   MArray<float,3> B;
   B = convolve(A, cderiv1(2));
   //cout << B << endl;
   LTL_ASSERT_( (B(5,4,3)==1 && B(5,6,3)==-1), "3-D MArray convolve with contiguous memory failed.");
   B(5,4,3) = B(5,6,3)=0;
   LTL_ASSERT_( noneof(B), "3-D MArray convolve with contiguous memory failed.");

   MArray<float,3> C(10,10,6);
   C = 0.0f;
   C(Range(3,8), Range::all(), Range(2,4)) = convolve(A(Range(3,8), Range::all(), Range(2,4)), cderiv1(2));
   //cout << C << endl;
   LTL_ASSERT_( (C(5,4,3)==1 && C(5,6,3)==-1), "2-D MArray convolve with non-contiguous memory failed.");
   C(5,4,3) = C(5,6,3) = 0;
   LTL_ASSERT_( noneof(C), "2-D MArray convolve with non-contiguous memory failed.");

   C = 0.0f;
   C(Range(3,8), Range(3,8), 3) = convolve(A(Range(3,8), Range(3,8), 3), cderiv1(2));
   //cout << C << endl;
   LTL_ASSERT_( (C(5,4,3)==1 && C(5,6,3)==-1), "2-D MArray convolve with non-contiguous memory slice failed.");
   C(5,4,3) = C(5,6,3) = 0;
   LTL_ASSERT_( noneof(C), "2-D MArray convolve with non-contiguous memory slice failed.");
}

void test_with_index()
{
   cerr << "Testing MArray convolution expression with index iterators ..." << endl;

   MArray<float,1> A(10), B(10);
   A = 0.0f;
   B = 0.0f;

   B = convolve( indexPosFlt(A,1), cderiv1(1) );
   //cout << B << endl;
   LTL_ASSERT_( B(1)==0.0f && B(10)==0.0f && allof(B(Range(2,9))==2.0f), "Convolve with index iter in 1-D failed." );

   MArray<float,2> AA(10,10), BB(10,10);
   AA = 0.0f;
   BB = 0.0f;

   BB = convolve( indexPosFlt(AA,1), cderiv1(1) );
   //cout << BB << endl;
   LTL_ASSERT_( allof(BB(1,Range::all())==0.0f) && allof(BB(10,Range::all())==0.0f) && allof(BB(Range(2,9),Range::all())==2.0f), "Convolve with index iter in 2-D failed." );

   BB = 0.0f;
   BB = convolve( indexPosFlt(AA,2), cderiv1(2) );
   //cout << BB << endl;
   LTL_ASSERT_( allof(BB(Range::all(),1)==0.0f) && allof(BB(Range::all(),10)==0.0f) && allof(BB(Range::all(),Range(2,9))==2.0f), "Convolve with index iter in 2-D failed." );
   BB = 0.0f;

   BB = convolve( indexPosFlt(AA,2), cderiv1(1) );
   //cout << BB << endl;
   LTL_ASSERT_( allof(BB==0.0f), "Convolve with index iter in 2-D failed." );
}

void test_grad()
{
   cerr << "Testing MArray gradients ..." << endl;
   MArray<float,2> A(10,10);
   MArray<FVector<float,2>,2> B(10,10);
   A = 0.0f;
   A(5,5) = 1.0f;
   B = convolve(A,grad2D());
   //cout << B << endl;
}

void test_gaussian()
{
   cerr << "Testing MArray convolved with Gaussian ..." << endl;
   MArray<double,1> A(13);
   A = 0;
   A(7) = 1.0;

   MArray<double,1> B(13);
   B = 0.0f;
   GaussianKernel1D G(1.0,3);
   B = convolve(A,G);
   cout << A << endl;
   cout << B << endl;
   cout << sum(A) << endl;
   cout << sum(B) << endl;
   LTL_ASSERT_(fabs(sum(A)-sum(B))<0.0001, "Convolve with Gaussian failed." );
}
