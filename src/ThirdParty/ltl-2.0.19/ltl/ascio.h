/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: ascio.h 537 2014-04-07 13:51:11Z snigula $
 * ---------------------------------------------------------------------
 *
 * Copyright (C)  Jan Snigula  <sniglua@usm.uni-muenchen.de>
 *                         Niv Drory    <drory@mpe.mpg.de>
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

#ifndef __LTL_ASCIO__
#define __LTL_ASCIO__

#include <ltl/config.h>

#include <string>
#include <fstream>
#include <iostream>
#include <vector>

#if defined HAVE_SSTREAM
#include <sstream>
#define ISTREAM std::istringstream
#define OSTREAM std::ostringstream
#elif defined HAVE_STRSTREAM
#include <strstream>
#define ISTREAM std::istrstream
#define OSTREAM std::ostrstream
#else
#error <sstream> or <strstream> needed!
#endif

#include <ltl/misc/exceptions.h>
#include <ltl/marray.h>

using std::string;
using std::fstream;
using std::streampos;
using std::vector;
using std::ios;
using std::cerr;
using std::endl;

namespace ltl {

/*! \addtogroup ma_ascii_io
*/
//@{

//! Columns based interface to an ASCII data file
class AscFile
{
   protected:
      //! The internal storage of the represented file
      string filename_;
      //! The filestream
      fstream in_;
      //! Internal storage of the position of the filestream
      streampos begin_;
      streampos end_;

      //! Internal storage of the number of rows in the file
      int rows_;
      //! Internal storage of the number of columns in the file
      int cols_;
      //! Internal storage of delimiter character
      char delim_;
      //! Internal storage of the comment string
      string comment_;
      //! Internal storage of next line to be read
      int currline_;

      //! Internal function to extract one column from a given string
      string readColumnFromLine__( const int col, const string & line );
      //! Internal function to replace one column in a line 
      string replaceColumnsInLine__( const int col1, const int col2, 
                                     string & line, const string & rep );
      //! Internal function used to read on line from the file stream
      bool readNextLine__( string & buf );
      //! Internal function to fast read the next column in a WS delimited file
      char* getNextColumn__( char* &str );


      void rewind__();

      bool eof__();

      //! Internal function used to count the number of rows
      void countRows__();
      //! Internal function used to count the number of columns
      void countCols__();

      template<class T>
      MArray<T,1> readSingleCol__( const int col, const int start=1, int nrows=-1 );
      MArray<float,1> readSingleFloatCol__( const int col, const int start=1, int nrows=-1 );
      MArray<int,1> readSingleIntCol__( const int col, const int start=1, int nrows=-1 );
      template<class T>
      MArray<T,2> readCols__( int first=0, int last=0, const int start=1, int nrows=-1 );
      MArray<float,2> readFloatCols__( int first=0, int last=0, const int start=1, int nrows=-1 );
      MArray<int,2> readIntCols__( int first=0, int last=0, const int start=1, int nrows=-1 );

   public:

      //! Constructs an AsciiFile object
      AscFile( const string &fname, const char delim=0,
               const string &comment="#" );

      AscFile( const string &fname, const int b, int e=-1, const char delim=0,
               const string &comment="#" );

      virtual ~AscFile();

      void close() { if(in_.is_open()) in_.close(); }
      bool readNextLine( string& buf ) { return readNextLine__(buf); }
      void rewind() { return rewind__(); }

      MArray<int,1> readIntColumn( const int col, const int start=1, int nrows=-1 ) throw(IOException);
      MArray<long long,1> readLongLongColumn( const int col, const int start=1, int nrows=-1 ) throw(IOException);

      MArray<float,1> readFloatColumn( const int col, const int start=1, int nrows=-1 ) throw(IOException);
      MArray<double,1> readDoubleColumn( const int col, const int start=1, int nrows=-1 ) throw(IOException);

      MArray<int,2> readIntColumns( const int first=0,
                                    const int last=0, const int start=1, int nrows=-1 ) throw(IOException);
      MArray<long long,2> readLongLongColumns( const int first=0,
                                               const int last=0, const int start=1, int nrows=-1 ) throw(IOException);

      MArray<float,2> readFloatColumns( const int first=0,
                                        const int last=0, const int start=1, int nrows=-1 ) throw(IOException);
      MArray<double,2> readDoubleColumns( const int first=0,
                                          const int last=0, const int start=1, int nrows=-1 ) throw(IOException);
      //! read arbitrary list of columns
      MArray<float,2> readFloatColumns( const int* cols, const int ncols, const int start=1, int nrows=-1 );

      //! High level interface for standard STL containers
      template< class T >
      int readColumn( const int col, T &cont, const int start=1, int nrows=-1 ) throw(IOException);
      //! High level interface for STL vectors
      template< class T >
      int readColumn( const int col, vector<T> &cont, const int start=1, int nrows=-1 ) throw(IOException);
      //! High level interface for C-style arrays
      template< class T >
      int readColumn( const int col, T* &cont, const int start=1, int nrows=-1 ) throw(IOException);
      // High level interface to a C-style array of C-Strings
      //  int readColumn( const int col, char** & );
      //  template< class T > int readColumn( const int col, MArray<T,1> &values );

      //! Replace one column and write result to ostream
      template< class T >
      void replaceColumn( const int col, const T & cont, ostream & os );

      //! Replace consecutive columns and write result to ostream
      template< class T >
      void replaceColumns( const int col1, const int col2, const T & cont, ostream & os );

      int getHeader( vector<string> &, bool keepcs=false );

      //! High level interface to ltl::AscFile::rows_
      int rows() throw(IOException);

      //! High level interface to ltl::AscFile::cols_
      int cols() throw(IOException);
};

// Declare specialization for vector<string>, gets rid of istringstream hack
// implemented in ascio.cpp
template<>
int AscFile::readColumn( const int col, vector<string> &cont, const int start, int nrows ) throw(IOException);


template< class T >
int AscFile::readColumn( const int col, T &cont, const int start, int nrows ) throw(IOException)
{
   string buff="";
   string b="";
   int count = 0;

   typename T::value_type tbuff;

   if( nrows == -1 ) {
      nrows = rows()-start+1;
   }

   if( currline_ > start ) // The current file pointer is past the first line to be read
      rewind__();

   while( start > currline_ )
      readNextLine__( buff ); // Seek to the first line to be read

   while( count<nrows )
   {
      if( !readNextLine__(buff) )
         break;

      b = readColumnFromLine__( col, buff );
      ISTREAM is(b.c_str());
      is >> tbuff;
      if( is.bad() )
         throw IOException( "Bad stream state!" );
      if( is.fail() )
         throw IOException( "Failed to read data from file!" );
      cont.push_back( tbuff );
      ++count;
   }

   return count;
}

template< class T >
int AscFile::readColumn( const int col, vector<T> &cont, const int start, int nrows ) throw(IOException)
{
   string buff="";
   string b="";
   int count = 0;

   T tbuff;

   if( nrows == -1 ) {
      nrows = rows()-start+1;
   }

   if( currline_ > start ) // The current file pointer is past the first line to be read
      rewind__();

   while( start > currline_ )
      readNextLine__( buff ); // Seek to the first line to be read

   if( (unsigned)nrows > cont.max_size() )
      throw IOException( "A memory error occured!" );

   cont.reserve(nrows);
   cont.clear();


   while( count<nrows )
   {
      if( !readNextLine__(buff) )
         break;

      b = readColumnFromLine__( col, buff );
      ISTREAM is(b.c_str());
      is >> tbuff;
      if( is.bad() )
         throw IOException( "Bad stream state!" );
      if( is.fail() )
         throw IOException( "Failed to read data from file!" );
      cont.push_back( tbuff );
      ++count;
   }

   return count;
}

template< class T >
int AscFile::readColumn( const int col, T* &cont, const int start, int nrows ) throw(IOException)
{
   string buff="";
   string b="";
   int count = 0;

   if( nrows == -1 ) {
      nrows = rows()-start+1;
   }

   if( currline_ > start ) // The current file pointer is past the first line to be read
      rewind__();

   while( start > currline_ )
      readNextLine__( buff ); // Seek to the first line to be read

   // Create the C-style array
   cont = new T[nrows];

   T tbuff;

   while( count<nrows )
   {
      if( !readNextLine__(buff) )
         break;

      b = readColumnFromLine__( col, buff );
      ISTREAM is(b.c_str());
      is >> tbuff;
      if( is.bad() )
         throw IOException( "Bad stream state!" );
      if( is.fail() )
         throw IOException( "Failed to read data from file!" );
      cont[count] = tbuff;
      ++count;
   }

   return count;
}

template< class T >
void AscFile::replaceColumn( const int col, const T & cont, ostream & os )
{
   // This is just an overloaded function for convenience
   replaceColumns( col, col, cont, os );
}

template< class T >
void AscFile::replaceColumns( const int col1, const int col2, const T & cont, ostream & os )
{
   string line = "";
   string buff = ""; //Test buffer

   typename T::const_iterator iter = cont.begin();
   typename T::const_iterator end_iter = cont.end();
   
   while( iter!=end_iter ) {

      line = "";

      if( eof__() )
      {
         rewind__();
         return;
      }
      getline( in_, line );

      string::size_type start = line.find( comment_, 0 );
      
      if( start != string::npos )
         buff= line.substr( 0, start ); // Found a comment in the line!!
      else 
         buff = line;
   
      if( delim_ == 0 ) 
         start = buff.find_first_not_of( " \t", 0 );
      else
         start = buff.find_first_not_of( delim_, 0 );
         
      if( start == string::npos ) {
         os << line+"\n"; // Empty line or just a comment
      } else {      
         os << replaceColumnsInLine__( col1, col2, line, *iter );
         ++iter;
      }
   }
}

//@}

/***********************************************************
 * Doxygen documentation block starts here
 **********************************************************/

/*! \class AscFile
  A high level interface class to read columns of data from an ASCII-file
  Supports arbitrary delimiter characters and whitespace seperated columns.
  Comments can begin at any point of a line. An arbitrary string can be used
  for the beginning of the comment.
*/

/*! \var int AscFile::rows_
  Stores internally the the number of rows in the file. This variable
  is initialized with 0. The first call to ltl::AscFile::rows() updates the
  variable with the correct number of rows.
*/

/*! \var int AscFile::cols_
  Stores internally the the number of columns in the file. This variable
  is initialized with 0. The first call to ltl::AscFile::cols() updates the
  variable with the correct number of columns.
*/

/*! \var char AscFile::delim_
  Stores internally the column delimiter. If the set to int(0) then
  it is assumed, that the columns are whitespace delimited, and any number
  of whitespace characters is treated as one delimiter.
*/

/*! \var string AscFile::comment_
  Defaults to '#'.
*/

/*! \fn string AscFile::readColumnFromLine__( const int col, const string & line )
  Extracts the col'th column from the string buff, and returns it as a string. 
  \throw IOException if an EOF is encountered before the requested colum
  was read.
  \throw IOException if the file cannot be opened.
*/

/*! \fn bool AscFile::readNextLine__( string & buf )
  Reads the next line from the input stream. The input stream is closed
  between subsequent calls, and the stream position is saved in the variable
  ltl::AscFile::filepos_.
*/

/*! \fn void AscFile::countRows__()
  Counts the rows in the file. Might be expensive on large files, so
  the function is not called before the information is requested by calling
  ltl::AscFile::rows().
*/

/*! \fn void AscFile::countCols__()
  Counts the cols in the file. Might be expensive on large files, so
  the function is not called before the information is requested by calling
  ltl::AscFile::cols().
*/

/*! \fn AscFile::AscFile( const string &fname, const char delim=0, const string &comment="#" )
  : in_(fname.c_str()), rows_(0), cols_(0), filename_(fname),
  delim_(delim), comment_(comment)
 
  Constructs an ltl::AscFile object corresponding to the file \e fname.
*/

/*! \fn template< class T > int AscFile::readColumn( const int col, T &cont, const int start=1, int nrows=-1 )
  User-interface class used to read the column col into the STL container
  cont.
 
  \warning Currently only container that provide push_back() are supported.
*/

/*! \fn template< class T > int AscFile::readColumn( const int col, T* &cont, const int start=1, int nrows=-1 )
  User-interface class used to read the column col into a C-style array.
*/

/* \fn int AscFile::readColumn( const int col, char** &cont)
  User-interface class used to read the column col into a C-style array
  of C-strings.
*/

/*! \fn int AscFile::rows() throw(IOException)
 
Returns the number of rows in the file.
 
*/

/*! \fn int AscFile::cols() throw(IOException)
 
Returns the number of columns in the file
 
*/

}

#endif // ACIIFILE_H
