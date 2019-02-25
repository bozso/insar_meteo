/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fftw.h 491 2011-09-02 19:36:39Z drory $
 * ---------------------------------------------------------------------
 *
 * Copyright (C)  Niv Drory <drory@mpe.mpg.de>
 *                Claus A. Goessl <cag@usm.uni-muenchen.de>
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

#ifndef __LTL_FFTW_H__
#define __LTL_FFTW_H__

#include <ltl/config.h>
#include <ltl/marray.h>

#include <complex>
#include <fftw3.h>

using std::complex;

namespace ltl {

// ====================================================================

/*! \file fftw.h FFTW3 interface for MArrays

   \ingroup fftw
*/

/*! \defgroup fftw FFTW3 interface for MArrays

  \ingroup marray
  \addindex fftw3
  \addindex fft

  Interface to call libfftw3 FFT routines on \c MArrays up to rank 3.

  Interface to fftw3 FFT library. Currently, each call to one of the
  FFT methods causes a new plan to be generated. This is highly inefficient,
  but safe. It guarantees that pointers to the data strorage of \c MArray do not
  escape by being stored in a plan that might survive the \c MArray object.

  All transforms leave the output array un-normalized. To normalize (divide by
  number of elements) call \c normalize().

  \code
   #include <ltl/fftw.h>

   // example of 1-dimensional FFT using fftw3 library:
   MArray<double, 1> in(size);
   MArray<std::complex<double>, 1> out(size);
   FourierTransform<double,1> FFT;

   in = 10.0 +
        20.0 * sin(indexPosDbl(in,1) / double(size) * 2.0 * M_PI * 3.0) +
        30.0 * cos(indexPosDbl(in,1) / double(size) * 2.0 * M_PI * 4.0);
   FFT.FFT_Real2Complex(in,out);
   FFT.normalize(out);
   out = merge(real(out * conj(out)) > 1e-9, out, 0);
   std::cout << out << std::endl;
  \endcode
*/


/*! FourierTransform class
 *
 * \ingroup fftw
 *
 * Interface to fftw3 FFT library. Currently, each call to one of the
 * FFT methods causes a new plan to be generated. This is highly inefficient,
 * but safe. It guarantees that pointers to the data strorage of \c MArray do not
 * escape by being stored in a plan that might survive the \c MArray object.
 *
 * All transforms leave the output array un-normalized. To normalize (divide by
 * number of elements) call \c normalize().
 *
 * \code
 *  #include <ltl/fftw.h>
 *
 *  // example of 1-dimensional FFT using fftw3 library:
 *  MArray<double, 1> in(size);
 *  MArray<std::complex<double>, 1> out(size);
 *  FourierTransform<double,1> FFT;
 *
 *  in = 10.0 +
 *       20.0 * sin(indexPosDbl(in,1) / double(size) * 2.0 * M_PI * 3.0) +
 *       30.0 * cos(indexPosDbl(in,1) / double(size) * 2.0 * M_PI * 4.0);
 *  FFT.FFT_Real2Complex(in,out);
 *  FFT.normalize(out);
 *  out = merge(real(out * conj(out)) > 1e-9, out, 0);
 *  std::cout << out << std::endl;
 * \endcode
 * */
template<typename T, int N>
class FourierTransform
{
   public:
      //! constructor
      FourierTransform();

      //! destructor
      virtual ~FourierTransform();

      //@{
      //! Forward and inverse fourier transform. Plan and execute the transform.  Output array is not normalized.
      void FFT(MArray<complex<T>,N> &A, MArray<complex<T>,N> &FFT_A);
      void iFFT(MArray<complex<T>,N> &FFT_A, MArray<complex<T>,N> &A);
      //@}

      //! Real to complex (forward) fourier transform. Plan and execute the transform. Output array is not normalized.
      void FFT_Real2Complex(MArray<T,N> &A, MArray<complex<T>,N> &FFT_A);
      //! Complex to real (inverse) fourier transform. Plan and execute the transform.  Output array is not normalized.
      void FFT_Complex2Real(MArray<complex<T>,N> &FFT_A, MArray<T,N> &A);

      //@{
      //! normalize the array by dividing by the number of elements.
      void normalize(MArray<complex<T>,N> &A);
      void normalize(MArray<T,N> &A);
      //@}

      //@{
      //! shift DC component to center of \c MArray.
      MArray<T,N> shiftDC(MArray<T,N> &A);
      MArray<complex<T>,N> shiftDC(MArray<complex<T>,N> &A);
      //@}

      //! dispose plan and call \c fftw_cleanup().
      void reset()
      {  mode = FFTW_ESTIMATE; fftw_cleanup();}

   protected:
      unsigned mode;
      fftw_plan plan;

      void execute();
};

/// \cond DOXYGEN_IGNORE
template<typename T, int N>
FourierTransform<T,N>::FourierTransform()
{
   mode = FFTW_ESTIMATE;
}

template<typename T, int N>
FourierTransform<T,N>::~FourierTransform()
{
   fftw_cleanup();
}

template<typename T, int N>
void FourierTransform<T,N>::FFT( MArray<complex<T>,N> &A,
                                 MArray<complex<T>,N> &FFT_A)
{
   LTL_ASSERT_(N < 4 && N > 0, "Array dimension must be 1, 2, or 3."); //Protect FFTW
   int extent[3];
   for (int i=0; i<N; ++i)
      extent[i] = A.length(i+1);

   plan = fftw_plan_dft(N, extent,
                        reinterpret_cast<fftw_complex*> (A.data()),
                        reinterpret_cast<fftw_complex*> (FFT_A.data()),
                        FFTW_FORWARD, mode);

   execute();
}

template<typename T, int N>
void FourierTransform<T,N>::iFFT( MArray<complex<T>,N> &FFT_A,
                                  MArray<complex<T>,N> &A)
{
   LTL_ASSERT_(N < 4 && N > 0, "Array dimension must be 1, 2, or 3."); //Protect FFTW
   int extent[3];
   for (int i=0; i<N; ++i)
      extent[i] = FFT_A.length(i+1);

   plan = fftw_plan_dft(N, extent,
                        reinterpret_cast<fftw_complex*> (FFT_A.data()),
                        reinterpret_cast<fftw_complex*> (A.data()),
                        FFTW_BACKWARD, mode);

   execute();
}

template<typename T, int N>
void FourierTransform<T,N>::FFT_Real2Complex( MArray<T,N> &A, MArray<complex<T>,N> &FFT_A)
{
   LTL_ASSERT_(N < 4 && N > 0, "Array dimension must be 1, 2, or 3."); //Protect FFTW
   int extent[3];
   for (int i=0; i<N; ++i)
      extent[i] = A.length(i+1);

   plan  = fftw_plan_dft_r2c(N, extent,
                             A.data(),
                             reinterpret_cast<fftw_complex*> (FFT_A.data()),
                             mode);

   execute();
}

template<typename T, int N>
void FourierTransform<T,N>::FFT_Complex2Real(MArray<complex<T>,N> &FFT_A, MArray<T,N> &A)
{
   LTL_ASSERT_(N < 4 && N > 0, "Array dimension must be 1, 2, or 3."); //Protect FFTW
   int extent[3];
   for (int i=0; i<N; ++i)
      extent[i] = FFT_A.length(i+1);

   plan = fftw_plan_dft_c2r(N, extent,
                            reinterpret_cast<fftw_complex*> (FFT_A.data()),
                            A.data(), mode);

   execute();
}

template<typename T, int N>
void FourierTransform<T,N>::normalize(MArray<complex<T>,N> &A)
{
   T factor;

   factor = static_cast<T> (A.nelements());
   A /= factor;
}

template<typename T, int N>
void FourierTransform<T,N>::normalize(MArray<T,N> &A)
{
   T factor;

   factor = static_cast<T> (A.nelements());
   A /= factor;
}

template<typename T, int N>
MArray<T,N> FourierTransform<T,N>::shiftDC(MArray<T,N> &A)
{
   int rowsHalf = static_cast<int> (A.length(1) / 2.0),
       colsHalf = static_cast<int> (A.length(2) / 2.0);
   MArray<T,N> tmpField(A.shape());

   for (int j = 1; j <= A.length(1); j++)
      for (int k = 1; k <= A.length(2); k++)
         tmpField((j + rowsHalf) % A.length(1), (k + colsHalf) % A.length(2))
               = A(j, k);

   return tmpField;
}

template<typename T, int N>
MArray<complex<T>,N> FourierTransform<T,N>::shiftDC(MArray<complex<T>,N> &A)
{
   int rowsHalf = static_cast<int> (A.length(1) / 2.0),
       colsHalf = static_cast<int> (A.length(2) / 2.0);
   MArray<complex<T>,N> tmpField(A.shape());

   for (int j = 1; j <= A.length(1); j++)
      for (int k = 1; k <= A.length(2); k++)
         tmpField((j + rowsHalf) % A.length(1), (k + colsHalf) % A.length(2))
               = A(j, k);

   return tmpField;
}

//Private
template<typename T, int N>
void FourierTransform<T,N>::execute()
{
   if (plan != NULL)
   {
      fftw_execute(plan);
      fftw_destroy_plan(plan);
   }
//   else
//      throw LTL::LTLException ("No plan established before calling FFTW3!");
}
/// \endcond

}

#endif // __LTL_FFTW_H__
