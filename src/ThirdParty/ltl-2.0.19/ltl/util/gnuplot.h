/*
Copyright (c) 2009 Daniel Stahlke

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

$Id: gnuplot.h 524 2013-06-17 15:18:43Z snigula $
*/

// modified by Niv Drory to work with ltl::MArray and
// not to depend on Boost::iostreams.

#ifndef GNUPLOT_IOSTREAM_H
#define GNUPLOT_IOSTREAM_H

#include <ltl/config.h>

#include <iostream>
#include <streambuf>
#include <string>
#include <cstdio>

#include <ltl/marray.h>
#include <ltl/fvector.h>

using ltl::MArray;

namespace ltl {

// ====================================================================

/*! \file gnuplot.h Gnuplot interface
 *
 *  \ingroup plotting
 */

/*! \defgroup plotting Gnuplot interface
 *
 * \ingroup marray
 * \addindex gnuplot
 * \addindex plot
 *
 * Simple interface to call gnuplot and send \c MArray and \c FVector objects.
 * An instance of a \c Gnuplot class encapsulates a pipe into gnuplot's stdin.
 * It exports a stream interface to the user, so that any gnuplot command can
 * be sent via \c operator<<. For convenience, functions handling the loops
 * over elements and the necessary formatting to plot elements of 1-D \c MArray
 * against index number, two 1-D \c MArray against each other, 2-D \c MArray
 * surfaces, and elements of any stl-iterator compatible container are provided.
 *
 * The follwing example code should be self-explanatory:
 *
 * \code
 *    MArray<float,1> A;
 *
 *    Gnuplot gp;  // start a gnuplot session and open pipe
 *    Gnuplot gp ("gnuplot-cmd"); // alternative command string
 *
 *    // plot elements of A against index number:
 *    gp << "plot '-' with lines\n";
 *    gp.send (A);
 *
 *    MArray<float,1> X, Y;
 *    // add a plot of Y vs. X to previous plot
 *    gp << "replot '-' with points\n";
 *    gp.send (X,Y);
 *
 *    MArray<float,2> S;
 *    // a surface plot of 2-D MArray S
 *    gp << "splot '-' with lines\n";
 *    gp.send (S);
 *
 *    std::vector<float> v;
 *    float *f;
 *    // plot an iterator range
 *    gp << "plot '-' with points\n";
 *    gp.send_iter (v.begin(), v.end());
 *    gp << "replot '-' with points\n";
 *    gp.send_iter (f, f+10);
 *
 *    double mx, my;
 *    int mb;
 *    gp.getMouse(mx, my, mb);
 * \endcode
*/



/// \cond DOXYGEN_IGNORE

/*
 *  Helper class to construct a strambuf from a pre-existing file descriptor
 */
class fdoutbuf: public std::streambuf
{
   public:
      // constructor
      fdoutbuf () : fd(0)
      {

      }

      virtual ~fdoutbuf ()
         { ::close(fd); }

      void set_fd(int fd_)
      {
         fd = fd_;
      }

   protected:
      // write one character
      virtual int_type overflow (int_type c)
      {
         if (c!=EOF)
         {
            char z = c;
            if (write (fd, &z, 1)!=1)
               return EOF;
         }
         return c;
      }

      // write multiple characters
      virtual std::streamsize xsputn (const char* s, std::streamsize num)
      {
         return write (fd, s, num);
      }

      int fd; // file descriptor
};

/*
 *  Helper class to construct a std::ostream and attach it to a pre-existing file descriptor
 *  via an \c fdoutbuf streambuf object
 */
class fdostream: public std::ostream
{
   public:
      fdostream () :
         std::ostream (0)//, buf (fd)
      {
         //rdbuf (&buf);
      }

      void init( int fd )
      {
         buf.set_fd(fd);
         rdbuf(&buf);
      }

   protected:
      fdoutbuf buf;
};
/// \endcond

/*! Gnuplot interface for \c MArrays and \c FVectors
 *
 * \ingroup plotting
 *
 * Simple interface to call gnuplot and send \c MArray and \c FVector objects.
 * An instance of a \c Gnuplot class encapsulates a pipe into gnuplot's stdin.
 * It exports a stream interface to the user, so that any gnuplot command can
 * be sent via \c operator<<. For convenience, functions handling the loops
 * over elements and the necessary formatting to plot elements of 1-D \c MArray
 * against index number, two 1-D \c MArray against each other, 2-D \c MArray
 * surfaces, and elements of any stl-iterator compatible container are provided.
 *
 * The follwing example code should be self-explanatory:
 *
 * \code
 *    MArray<float,1> A;
 *
 *    Gnuplot gp;  // start a gnuplot session and open pipe
 *    Gnuplot gp ("gnuplot-cmd"); // alternative command string
 *
 *    // plot elements of A against index number:
 *    gp << "plot '-' with lines\n";
 *    gp.send (A);
 *
 *    MArray<float,1> X, Y;
 *    // add a plot of Y vs. X to previous plot
 *    gp << "replot '-' with points\n";
 *    gp.send (X,Y);
 *
 *    MArray<float,2> S;
 *    // a surface plot of 2-D MArray S
 *    gp << "splot '-' with lines\n";
 *    gp.send (S);
 *
 *    std::vector<float> v;
 *    float *f;
 *    // plot an iterator range
 *    gp << "plot '-' with points\n";
 *    gp.send_iter (v.begin(), v.end());
 *    gp << "replot '-' with points\n";
 *    gp.send_iter (f, f+10);
 *
 *    double mx, my;
 *    int mb;
 *    gp.getMouse(mx, my, mb);
 * \endcode
 *
 * \c Gnuplot::send() works with \c MArrays up to dimension 2, \c FVectors.
 * For stdlib containers use \c send with any valid iterator range, i.e. \c send(v.begin(),v.end())
 *
 * \c Gnuplot inherits from \c std::ostream, so any gnuplot commands and data can be sent to \c Gnuplot
 * via standard streamio operations, \c operator<<, \c setprecision(), etc.
 *
 */
class Gnuplot : public fdostream
{
   public:
      /*!
       *  Construct with command to launch gnuplot. This will call gnuplot and open a pipe.
       */
      Gnuplot(const std::string cmd = "");
      ~Gnuplot();

      /*!
       *  Wait for mouse event in gnuplot window and return corrdinates and button
       */
      void getMouse(double &mx, double &my, int &mb);

      /*!
       * Any \c iterator range
       */
      template <class T>
      Gnuplot& send_iter(T p, const T& last);

      /*!
       *  1-Dimensional \c MArray
       */
      template <class T>
      Gnuplot& send(const ltl::MArray<T,1>& A);

      /*!
       *  Plot 2 \c MArrays against each other
       */
      template <class T1, class T2>
      Gnuplot& send(const ltl::MArray<T1,1>& X, const ltl::MArray<T2,1>& Y);

      /*!
       *  2-Dimensional \c MArray
       */
      template <class T>
      Gnuplot& send(const ltl::MArray<T,2>& A);

      /*!
       *  2-Dimensional \c MArray to be used with "plot '-' matrix with image"
       *  Note: T could be either a scalar or a blitz::TinyVector.
       */
      template <class T>
      Gnuplot& sendAsImage(const ltl::MArray<T,2> &A);

      void interactive( const string& pname );

   protected:
      /*!
       *  Handles container elements
       */
      template <class T>
      Gnuplot& send(const T& x)
      {
         sendEntry(x);
         return *this;
      }

      template <class T, int N>
      void sendEntry(const ltl::FVector<T,N>& v)
      {
         for (int i=0; i<N; i++)
            sendEntry(v[i]);
      }

      template <class T, int N>
      void sendEntry(const ltl::FixedVector<T,N>& v)
      {
         for (int i=0; i<N; i++)
            sendEntry(v[i]);
      }

      template <class T>
      void sendEntry(const T v)
      {
         *this << v << " ";
      }

      template <class T, class U>
      void sendEntry(const std::pair<T,U>& v)
      {
         sendEntry(v.first);
         sendEntry(v.second);
      }

      void allocReader();

      FILE *pout;
      std::string pty_fn;
      FILE *pty_fh;
      int master_fd, slave_fd;
      bool debug_messages;
};


/*!
 * Any \c iterator range
 */
template <class T>
Gnuplot& Gnuplot::send_iter(T p, const T& last)
{
   while(p != last)
   {
      send(*p) << std::endl;
      ++p;
   }
   *this << "e" << std::endl;
   return *this;
}

/*!
 *  1-Dimensional \c MArray
 *  Will be plotted against index number.
 */
template <class T>
Gnuplot& Gnuplot::send(const ltl::MArray<T,1>& A)
{
   MArray<T,1> ind(A.shape());
   ind = indexPosInt(A,1);

   return send(ind,A);
}

/*!
 *  2-Dimensional \c MArray
 */
template <class T>
Gnuplot& Gnuplot::send(const ltl::MArray<T,2> &A)
{
   for (int i=A.minIndex(2); i<=A.maxIndex(2); ++i)
   {
      for (int j=A.minIndex(1); j<=A.maxIndex(1); ++j)
         send(A(j, i)) << "\n";
      *this << "\n";
   }
   *this << "e" << std::endl;
   return *this;
}

/*!
 *  2-Dimensional \c MArray to be used with "plot '-' matrix with image"
 *  Note: T could be either a scalar or a blitz::TinyVector.
 */
template <class T>
Gnuplot& Gnuplot::sendAsImage(const ltl::MArray<T,2> &A)
{
   for (int i=A.minIndex(2); i<=A.maxIndex(2); ++i)
   {
      for (int j=A.minIndex(1); j<=A.maxIndex(1); ++j)
         *this << A(j,i) << " ";
      *this << "\n";
   }
   *this << "e" << endl << "e" << endl;
   return *this;
}

/*!
 *  Plot 2 \c MArrays against each other
 */
template <class T1, class T2>
Gnuplot& Gnuplot::send(const ltl::MArray<T1,1>& X, const ltl::MArray<T2,1>& Y)
{
   CHECK_CONFORM(X,Y);
   for (int i=X.minIndex(1), j=Y.minIndex(1); i<=X.maxIndex(1); ++i, ++j)
   {
      send(X(i)) << " ";
      send(Y(j)) << std::endl;
   }
   *this << "e" << std::endl;
   return *this;
}

}

#endif // GNUPLOT_IOSTREAM_H
