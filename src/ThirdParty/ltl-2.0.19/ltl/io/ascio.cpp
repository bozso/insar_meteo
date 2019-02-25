/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: ascio.cpp 537 2014-04-07 13:51:11Z snigula $
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

#include <ltl/ascio.h>

#include <algorithm>

namespace ltl {

AscFile::AscFile( const string &fname, const char delim,
                  const string &comment )
      : filename_(fname), in_(fname.c_str(), ios::in),
        begin_(0), end_(-1), rows_(0), cols_(0), delim_(delim), 
        comment_(comment), currline_(1)
{
   if( !in_.good() )
      throw IOException( "Cannot open file '"+filename_+"'!" );
}


AscFile::AscFile( const string &fname, const int b, int e,
                  const char delim,
                  const string &comment )
      : filename_(fname), in_(fname.c_str(), ios::in),
        begin_(0), end_(-1), rows_(0), cols_(0), delim_(delim), 
        comment_(comment), currline_(1)
{
   if( !in_.good() )
      throw IOException( "Cannot open file '"+filename_+"'!" );

   if( b < 1 || ( b > e && e != -1 ) )
      throw IOException( "Irregular lines parameters in file '"+filename_+"'!" );

   bool foundline=false;
   string buff;
   int count=1;

   if( b != 1 )
   {
      while( readNextLine__(buff) )
         if( ++count > b )
         {
            foundline=true;
            break;
         }

      if( !foundline )
      {
         OSTREAM err;
         err << "No such line: " << b;
         throw IOException( err.str() );
      }
      // Now we fool reset the linecounter to the beginning of the current line
      //	    cerr << "buff is " << buff << endl;
      int l =  buff.length()+1;
      in_.seekg( -l, ios::cur );
      --count;

      begin_ = in_.tellg();
   }

   foundline=false;

   if( e != -1 )
   {
      while( readNextLine__(buff) )
         if ( ++count > e )
         {
            foundline=true;
            break;
         }

      if( !foundline )
         // Last line cannot be found using EOF
         end_ = -1;
      else
         end_ = in_.tellg();
   }
   rewind__();
}

AscFile::~AscFile()
{
   if( in_.is_open() )
      in_.close();
}

int AscFile::rows() throw(IOException)
{
   if( rows_ == 0 )
      countRows__();
   return rows_;
}

int AscFile::cols() throw(IOException)
{
   if( cols_ == 0 )
      countCols__();
   return cols_;
}


/*! Try to read the next available line from the input filestream.
  Returns false if no more lines can be read.
  \throw ltl::IOException() if the file cannot be opened.
*/
bool AscFile::readNextLine__( string &buff )
{

   buff = ""; // Empty the buffer

   if( eof__() )
   {
      rewind__();
      return false;
   }
   getline( in_, buff );

   if( delim_ == 0 )
   { // Eat leading whitespaces
      unsigned int pos=0;
      while( pos <= buff.length() )
      {
         if( !isspace( buff[pos++] ) )
            break;
      }
      buff = buff.substr( --pos, string::npos );
   }

   string::size_type start = buff.find( comment_, 0 );

   if( start != string::npos )
   { // Found a comment in the line!!
      buff = buff.substr( 0, start );
   }

   // We finished all modifications to buff at this point
   if( buff == "" ) // Line is empty... skip it
      return readNextLine__( buff );

   // At this point a valid line was successfully read 
   // Increase the line counter
   ++currline_;

   return true;
}

string AscFile::readColumnFromLine__( const int col, const string &line )
{
   if( delim_ == 0 )
   {
      string buff = "";
      char *tmpstr, *tmpval;
      int colcount = 0;

      // Make a copy of the line to work on
      tmpstr = new char[line.size()+1];
      strncpy (tmpstr, line.c_str(), line.size()+1);
      char *r = tmpstr; // Pointer to the original allocation

      while( colcount++ != col )
      {
         tmpval = getNextColumn__( tmpstr );
         
         if( colcount == col ) {               
            buff = string( tmpval );
         }
      }

      delete[] r;
      
      return buff;
   }
   else
   {
      typedef string::size_type ST;
      int colcount = 0;
      ST start = 0;
      ST end = 0;

      if( col == 1 )
      {
         /* The line starts with a delimiter,
            so the first col is empty... */
         if( line[0] == delim_ )
            return "";
         else
         {
            /* We must treat the first column as a special case, to
               prevent that the first char of the first column is deleted */
            end = line.find( delim_, start );
            return line.substr( start, end );
         }
      }

      while( ++colcount != col )
      {
         start = line.find( delim_, end );
         if( start == string::npos )
            throw IOException( "Column does not exist in file'"+filename_+"'!" ); // no such column!
         end = start+1;
      }

      end = line.find( delim_, start+1 );
      return line.substr( start+1, (end-start-1) );
   }
}

string AscFile::replaceColumnsInLine__( const int col1, const int col2, string & line, const string & rep )
{
   typedef string::size_type ST;
   
   int colcount = 0;
   ST start = 0;
   ST end = 0;
   
   if( col1 > col2 ) {
      throw IOException( "Start col > end col in file '"+filename_+"'!" ); // no such column!
   }

   string d="";
   
   if( delim_ == 0 ) 
      d = " \t";
   else
      d=delim_;
   
   if( col2 > cols() ) // Easy hack to replace to end of line
      end = string::npos;
   
   if( col1 == 1 ) 
      start = line.find_first_not_of( d );
   else {
      ST t1 = line.find_first_not_of( d, 0 );
      ST t2 = line.find_first_of( d, t1 );
      colcount = 1;
      while( colcount != col1 ) {
         t1 = line.find_first_not_of( d, t2 );
         t2 = line.find_first_of( d, t1 );
         ++colcount;
      }
      
      start = t1;
   }
      
   if( col1 == col2 )
      end = line.find_first_of( d, start );
   else if( !end ) {

      ST t1 = start;
      ST t2 = line.find_first_of( d, t1 );
      while( colcount != col2 ) {
         t1 = line.find_first_not_of( d, t2 );
         t2 = line.find_first_of( d, t1 );
         ++colcount;
      }
      
      end = t2;
   }

   return line.replace( start, end-start, rep )+"\n"; // Possible bug here!
     
}

char* AscFile::getNextColumn__( char* &str )
{
   char *val;

   val = strsep( &str, " \t" );
   // Skip empty fields
   while( val != 0 && *val == '\0' ) 
      val = strsep( &str, " \t" );
   if( val == 0 )
      throw IOException( "Not enough columns in file '"+filename_+"'!" );

   return val;
}

void AscFile::rewind__()
{
   in_.clear();
//   in_.seekg(begin_, ios::beg);
   in_.seekg(0, ios::beg);
   currline_=1;
}

bool AscFile::eof__()
{
   if( end_ == streampos(-1) )
      return in_.eof();
   else
      return ( in_.eof() || (in_.tellg()>=end_) );
}

void AscFile::countCols__()
{
   typedef string::size_type ST;
   string buff;

   // Remember the current position in the file
   streampos fp = in_.tellg();
   int cl = currline_;
   rewind__();

   if( !readNextLine__( buff ) )
   {
      cols_ = 0;
      return;
   }

   int colcount = 0;
   ST start = 0;

   if( delim_ == 0 )
   { 
     ISTREAM is( buff.c_str() );
      string devnull;
      while( is >> devnull )
      {
         if( devnull != "" )
            ++colcount;
      }
   }
   else
   {
      while( start != string::npos )
      {
         start = buff.find( delim_, start );
         if( start != string::npos )
         {
            ++start;
            ++colcount;
         }
         else
            break;
      }
      ++colcount;
   }
   cols_ = colcount;

   // Reposition the input stream

   in_.clear();
   in_.seekg( fp );
   currline_ = cl;

   return;
}

void AscFile::countRows__()
{
   streampos fp = in_.tellg();
   int cl = currline_; // Save current state
   rewind__();

   string buff;
   int rowcount = 0;
   while( readNextLine__(buff) )
   {
      ++rowcount;
   }

   rows_ = rowcount;

   // Rewind the file stream

   in_.clear();
   in_.seekg( fp );
   currline_ = cl; // Restore current state

}

template<class T>
MArray<T,1> AscFile::readSingleCol__( const int col, const int start, int nrows )
{
   string buff, b;
   int count = 0;
   T tbuff;

   if( nrows == -1 ) {
      nrows = rows()-start+1;
   }

   MArray<T,1> a( nrows );

   if( currline_ > start ) // The current file pointer is past the first line to be read
      rewind__();

   while( start > currline_ )
      readNextLine__( buff ); // Seek to the first line to be read

   while( count<nrows )
   {
      if( !readNextLine__(buff) )
         break;

      b = readColumnFromLine__( col, buff );
      ISTREAM is( b.c_str() );
      is >> tbuff;
      if( is.bad() )
         throw IOException( "Bad stream state in file '"+filename_+"'!" );
      if( is.fail() )
         throw IOException( "Failed to read data from file '"+filename_+"'!" );
      a( ++count ) = tbuff;
   }

   return a;
}

// Fast specialization
template<>
MArray<float,1> AscFile::readSingleCol__( const int col, const int start, int nrows )
{

   string buff, b;
   int count = 0;

   if( nrows == -1 ) {
      nrows = rows()-start+1;
   }

   MArray<float,1> a( nrows );


   if( currline_ > start ) // The current file pointer is past the first line to be read
      rewind__();

   while( start > currline_ )
      readNextLine__( buff ); // Seek to the first line to be read

   while( count<nrows )
   {
      if( !readNextLine__(buff) )
         break;
      b = readColumnFromLine__( col, buff );
      a( ++count ) = float(::atof( b.c_str() ));
   }

   return a;
}

// Fast specialization
template<>
MArray<int,1> AscFile::readSingleCol__( const int col, const int start, int nrows )
{
   string buff, b;
   int count = 0;

   if( nrows == -1 ) {
      nrows = rows()-start+1;
   }

   MArray<int,1> a( nrows );

   if( currline_ > start ) // The current file pointer is past the first line to be read
      rewind__();

   while( start > currline_ )
      readNextLine__( buff ); // Seek to the first line to be read

   while( count<nrows )
   {
      if( !readNextLine__(buff) )
         break;

      b = readColumnFromLine__( col, buff );
      a( ++count ) = ::atoi( b.c_str() );
   }

   return a;
}

// Fast specialization
template<>
MArray<long long,1> AscFile::readSingleCol__( const int col, const int start, int nrows )
{
   string buff, b;
   int count = 0;

   if( nrows == -1 ) {
      nrows = rows()-start+1;
   }

   MArray<long long,1> a( nrows );

   if( currline_ > start ) // The current file pointer is past the first line to be read
      rewind__();

   while( start > currline_ )
      readNextLine__( buff ); // Seek to the first line to be read

   while( count<nrows )
   {
      if( !readNextLine__(buff) )
         break;

      b = readColumnFromLine__( col, buff );
      a( ++count ) = ::strtoll( b.c_str(), 0, 10 );
   }

   return a;
}

template<class T>
MArray<T,2> AscFile::readCols__( int first, int last, const int start, int nrows )
{
   string buff, b, tmp="";
   T tbuff;

   if( nrows == -1 ) {
      nrows = rows()-start+1;
   }

   if( first==0 )
      first = 1;

   if( last==0 )
      last = cols();

   MArray<T,2> a( last-first + 1, nrows );

   if( currline_ > start ) // The current file pointer is past the first line to be read
      rewind__();

   while( start > currline_ )
      readNextLine__( buff ); // Seek to the first line to be read

   int count = 1;
   if( delim_ == 0 )
   {
      // faster version for whitespace delimiters
      while( count<=nrows )
      {
         if( !readNextLine__(buff) )
            break;

         int colcount = 1;
         ISTREAM is( buff.c_str() );
         while( colcount++ < first )
            is >> tmp;
         for( int i = first; i<= last; i++ )
         {
            is >> tbuff;
            a( i-first+1, count ) = tbuff;
         }
         if( is.bad() )
            throw IOException( "Bad stream state in file '"+filename_+"'!" );
         if( is.fail() )
            throw IOException( "Failed to read data from line: " + buff );
         count++;
      }      
   }
   else
   {
      while( count<=nrows )
      {
         if( !readNextLine__(buff) )
            break;

         for( int i = first; i<= last; i++ )
         {
            b = readColumnFromLine__( i, buff );
            ISTREAM is( b.c_str() );
            is >> tbuff;
            if( is.bad() )
               throw IOException( "Bad stream state!" );
            if( is.fail() )
               throw IOException( "Failed to read data from line: " + buff );
            a( i-first+1, count ) = tbuff;
         }
         count++;
      }
   }
   return a;
}

// Fast specialization
template<>
MArray<float,2> AscFile::readCols__( int first, int last, const int start, int nrows )
{
   string buff, b;
   char *tmpstr, *tmpval;
   float tbuff;

   if( nrows == -1 ) {
      nrows = rows()-start+1;
   }

   if( first==0 )
      first = 1;

   if( last==0 )
      last = cols();

   MArray<float,2> a( last-first+1, nrows );

   if( currline_ > start ) // The current file pointer is past the first line to be read
      rewind__();

   while( start > currline_ )
      readNextLine__( buff ); // Seek to the first line to be read

   int count = 1;
   if( delim_ == 0 )
   {
      // faster version for whitespace delimiters
      while( count<=nrows )
      {
         if( !readNextLine__(buff) )
            break;

         tmpstr = new char[buff.size()+1];
         strncpy (tmpstr, buff.c_str(), buff.size()+1);
         char *r = tmpstr;  // Pointer to the original allocation

         for( int i = 1; i<= last; i++ )
         {
            tmpval = getNextColumn__( tmpstr );

            if( i >= first ) {               
               a(i-first+1,count) = ::atof( tmpval );
            }
         }

         delete[] r;

         ++count;
      }      
   }
   else
   {
      while( count<=nrows )
      {
         if( !readNextLine__(buff) )
            break;

         for( int i = first; i<= last; i++ )
         {
            b = readColumnFromLine__( i, buff );
            ISTREAM is( b.c_str() );
            is >> tbuff;
            if( is.bad() )
               throw IOException( "Bad stream state in file '"+filename_+"'!" );
            if( is.fail() )
               throw IOException( "Failed to read data from line: " + buff );
            a( i-first+1, count ) = tbuff;
         }
         count++;
      }
   }
   return a;
}

// Fast specialization
template<>
MArray<int,2> AscFile::readCols__( int first, int last, const int start, int nrows )
{
   string buff, b;
   char *tmpstr, *tmpval;
   int tbuff;

   if( nrows == -1 ) {
      nrows = rows()-start+1;
   }

   if( first==0 )
      first = 1;

   if( last==0 )
      last = cols();

   MArray<int,2> a( last - first + 1, nrows );

   if( currline_ > start ) // The current file pointer is past the first line to be read
      rewind__();

   while( start > currline_ )
      readNextLine__( buff ); // Seek to the first line to be read

   int count = 1;
   if( delim_ == 0 )
   {
      // faster version for whitespace delimiters
      while( count<=nrows )
      {
         if( !readNextLine__(buff) )
            break;

         tmpstr = new char [buff.size()+1];
         strncpy (tmpstr, buff.c_str(), buff.size()+1);
         char *r = tmpstr;  // Pointer to the original allocation

         for( int i = 1; i<= last; i++ )
         {
            tmpval = getNextColumn__( tmpstr );

            if( i >= first ) {
               a(i-first+1,count) = ::atoi( tmpval );
            }
         }

         delete[] r;

         ++count;
      }      
   }
   else
   {
      while( count<=nrows )
      {
         if( !readNextLine__(buff) )
            break;

         for( int i = first; i<= last; i++ )
         {
            b = readColumnFromLine__( i, buff );
            ISTREAM is( b.c_str() );
            is >> tbuff;
            if( is.bad() )
               throw IOException( "Bad stream state in file '"+filename_+"'!" );
            if( is.fail() )
               throw IOException( "Failed to read data from line: " + buff );
            a( i-first+1, count ) = tbuff;
         }
         count++;
      }
   }
   return a;
}

// Fast specialization
template<>
MArray<long long,2> AscFile::readCols__( int first, int last, const int start, int nrows )
{
   string buff, b;
   char *tmpstr, *tmpval;
   long long tbuff;

   if( nrows == -1 ) {
      nrows = rows()-start+1;
   }

   if( first==0 )
      first = 1;

   if( last==0 )
      last = cols();

   MArray<long long,2> a( last - first + 1, nrows );

   if( currline_ > start ) // The current file pointer is past the first line to be read
      rewind__();

   while( start > currline_ )
      readNextLine__( buff ); // Seek to the first line to be read

   int count = 1;
   if( delim_ == 0 )
   {
      // faster version for whitespace delimiters
      while( count<=nrows )
      {
         if( !readNextLine__(buff) )
            break;

         tmpstr = new char [buff.size()+1];
         strncpy (tmpstr, buff.c_str(), buff.size()+1);
         char *r = tmpstr;  // Pointer to the original allocation

         for( int i = 1; i<= last; i++ )
         {
            tmpval = getNextColumn__( tmpstr );

            if( i >= first ) {
               a(i-first+1,count) = ::strtoll( tmpval, 0, 10 );
            }
         }

         delete[] r;

         ++count;
      }
   }
   else
   {
      while( count<=nrows )
      {
         if( !readNextLine__(buff) )
            break;

         for( int i = first; i<= last; i++ )
         {
            b = readColumnFromLine__( i, buff );
            ISTREAM is( b.c_str() );
            is >> tbuff;
            if( is.bad() )
               throw IOException( "Bad stream state in file '"+filename_+"'!" );
            if( is.fail() )
               throw IOException( "Failed to read data from line: " + buff );
            a( i-first+1, count ) = tbuff;
         }
         count++;
      }
   }
   return a;
}

MArray<int,1> AscFile::readIntColumn( const int col, const int start, int nrows ) throw(IOException)
{
   return readSingleCol__<int>(col,start,nrows);
}

MArray<long long,1> AscFile::readLongLongColumn( const int col, const int start, int nrows ) throw(IOException)
{
   return readSingleCol__<long long>(col,start,nrows);
}

MArray<float,1> AscFile::readFloatColumn( const int col, const int start, int nrows ) throw(IOException)
{
   return readSingleCol__<float>(col, start, nrows);
//   return readSingleFloatCol__(col);
}

MArray<double,1> AscFile::readDoubleColumn( const int col, const int start, int nrows ) throw(IOException)
{
   return readSingleCol__<double>(col, start, nrows);
}

MArray<int,2> AscFile::readIntColumns( const int first,
                                       const int last, const int start, int nrows ) throw(IOException)
{
   return readCols__<int>( first, last, start, nrows );
}

MArray<long long,2> AscFile::readLongLongColumns( const int first,
                                                  const int last, const int start, int nrows ) throw(IOException)
{
   return readCols__<long long>( first, last, start, nrows );
}

MArray<float,2> AscFile::readFloatColumns( const int first,
      const int last, const int start, int nrows ) throw(IOException)
{
   return readCols__<float>( first, last, start, nrows );
//   return readFloatCols__( first, last );
}

MArray<double,2> AscFile::readDoubleColumns( const int first,
      const int last, const int start, int nrows ) throw(IOException)
{
   return readCols__<double>( first, last, start, nrows );
}

int AscFile::getHeader( vector<string> &v, bool keepcs )
{
   /* Read the comment header from the beginning of the file 
    and return the number of lines read */

   string buff; // Empty the buffer

   rewind__(); // Rewind the filepointer to the begin

   while( !eof__() )
   {
      
      buff = "";

      getline( in_, buff ); // Read one line

      // Strip leading whitespace
      unsigned int pos=0;
      while( isspace( buff[pos] ) && pos <= buff.length() )
         ++pos;

      if( pos )
         buff = buff.substr( pos, string::npos );
      
      // If line empty continue
      if( buff == "" ) 
         continue;

      if( buff.length() >= comment_.length() )
         if( buff.substr( 0, comment_.length() ) != comment_ ) // Found a not empty line 
            break;

      if( !keepcs ) // Strip comment delimiter
         buff = buff.substr( comment_.length(), string::npos );

      // Delete trailing newline
      buff = buff.substr(0, buff.rfind( "\n" ) );
      
      // We came here so this is a valid comment -> add it to the vector
      v.push_back( buff );
   }

   rewind__(); // Rewind the filepointer to the begin

   return v.size();
}

// Specialization for vector<string>, gets rid of istringstream hack

template<>
int AscFile::readColumn( const int col, vector<string> &cont, const int start, int nrows ) throw(IOException)
{
   string buff;
   int count = 0;

   if( nrows == -1 ) {
      nrows = rows()-start+1;
   }

   if( currline_ > start ) // The current file pointer is past the first line to be read
      rewind__();

   while( start > currline_ )
      readNextLine__( buff ); // Seek to the first line to be read

   if( size_t(nrows) > cont.max_size() )
      throw IOException( "A memory error occured!" );

   cont.reserve(nrows);
   cont.clear();

   while( count<nrows )
   {
      if( !readNextLine__(buff) )
         break;

      cont.push_back( readColumnFromLine__( col, buff ) );
      ++count;
   }

   return count;
}

struct col__ {
      int index;
      int column;
};

struct sortByCol__ {
      bool operator() ( const col__ &lhs, const col__ &rhs ) {
         return rhs.column > lhs.column;
      }
};

MArray<float,2> AscFile::readFloatColumns( const int* _cols, const int ncols, const int start, int nrows )
{

   // Sort the cols array in ascending order but keep the indeces for the output MArray

   std::vector< struct col__ > columns;

   for( int i = 0; i<ncols; ++i ) {
      struct col__ t = { i+1, _cols[i] };
      columns.push_back( t );
   }

   std::sort( columns.begin(), columns.end(), sortByCol__() );

   string buff;
   char *tmpstr, *tmpval;
   
   if( delim_ != 0 )
      throw IOException( "Can only read whitespace delims in file '"+filename_+"'!" );

   if( _cols[0] < 0 || _cols[ncols-1] > cols() )
      throw IOException( "Incompatible column range in file '"+filename_+"'!" );

   if( nrows == -1 ) {
      nrows = rows()-start+1;
   }

   MArray<float,2> a( ncols, nrows );

   if( currline_ > start ) // The current file pointer is past the first line to be read
      rewind__();

   while( start > currline_ )
      readNextLine__( buff ); // Seek to the first line to be read
   

   int count = 1;

   // faster version for whitespace delimiters
   while( count<=nrows )
   {
      if( !readNextLine__(buff) )
         break;

      tmpstr = new char [buff.size()+1];
      strncpy (tmpstr, buff.c_str(), buff.size()+1);
      char *r = tmpstr;  // Pointer to the original allocation

      int j = 0;
      for( int i = 1; i<= columns[ncols-1].column; ++i )
      {
         
         tmpval = getNextColumn__( tmpstr );

         if( i == columns[j].column )
            a( columns[j++].index, count ) = ::atof( tmpval );
      }

      delete[] r;
         
      ++count;
   }
   
   return a;
}

}

