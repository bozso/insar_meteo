/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: memblock.h 496 2011-11-16 21:10:03Z drory $
 * ---------------------------------------------------------------------
 *
 * Copyright (C)  Niv Drory <drory@mpe.mpg.de>
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


#ifndef __LTL_MEMBLOCK__
#define __LTL_MEMBLOCK__

#include <ltl/config.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// mmap file
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>


#if defined(LTL_THREADSAFE_MEMBLOCK) && !defined(HAVE_ATOMIC_BUILTINS)
#include <pthread.h>
#endif


namespace ltl {

// The Class SynchronizedCounter provides a thread-synchronized
// simple integer counter used for reference counting in threaded
// environments like OpenMP.
// Only increment and decrement are protected by a mutex.
//
// Currently, the counter uses either a posix mutex, or gcc builtin
// atomic increment and decrement if LTL_THREADSAFE_MEMBLOCK is
// defined. Otherwise, no locking is performed.
//
class SynchronizedCounter
{
   public:
      SynchronizedCounter()
         : counter_(0)
      {
         initMutex();
      }
      
      SynchronizedCounter( const int n )
         : counter_(n)
      {
         initMutex();
      }

      virtual ~SynchronizedCounter()
      {
         destroyMutex();
      }
      
      int IncAndFetch()
      {
         int c;
#if defined(LTL_THREADSAFE_MEMBLOCK)
#  if !defined(HAVE_ATOMIC_BUILTINS)
         pthread_mutex_lock( &mutex_ );
         c = ++counter_;
         pthread_mutex_unlock( &mutex_ );
#  else
         c = __sync_add_and_fetch( &counter_, 1 );
#  endif
#else
          // no synchronization
         c = ++counter_;
#endif
         return c;
      }

      int DecAndFetch()
      {
         int c;
#if defined(LTL_THREADSAFE_MEMBLOCK)
#  if !defined(HAVE_ATOMIC_BUILTINS)
         pthread_mutex_lock( &mutex_ );
         c = --counter_;
         pthread_mutex_unlock( &mutex_ );
#  elif
         c = __sync_sub_and_fetch( &counter_, 1 );
#  endif
#else
         // no synchronization
         c = --counter_;
#endif
         return c;
      }

      int Counter() const
      {
         return counter_;
      }

   protected:
      void initMutex()
      {
#ifdef LTL_THREADSAFE_MEMBLOCK
#  if !defined(HAVE_ATOMIC_BUILTINS)
         pthread_mutex_init( &mutex_, NULL );
#  endif
#endif
      }

      void destroyMutex()
      {
#ifdef LTL_THREADSAFE_MEMBLOCK
#  if !defined(HAVE_ATOMIC_BUILTINS)
         pthread_mutex_destroy( &mutex_ );
#  endif
#endif
      }

   private:
      int counter_;
#ifdef LTL_THREADSAFE_MEMBLOCK
#  if !defined(HAVE_ATOMIC_BUILTINS)
         mutable pthread_mutex_t mutex_;
#  endif
#endif
};

   

// Class MemoryBlock provides a reference-counted block of memory.  This block
// may be referred to by multiple array objects.  The memory
// is automatically deallocated when the last referring object is destructed.
template<class T>
class MemoryBlock
{
   public:
      typedef T value_type;

      MemoryBlock()
         : data_(0), References_(0)
      { }

      MemoryBlock( const int items )
         : data_(new value_type[items]), References_(0)
      { 
#ifdef LTL_DEBUG_MEMORY_BLOCK
         cerr << "MemoryBlock " << this
              << "   Allocating : " << data_ << endl;
#endif
      }
      
      //! Create MemoryBlock from pre-allocated memory.
      MemoryBlock( value_type* restrict_ data )
         : data_(data), References_(0)
      { 
#ifdef LTL_DEBUG_MEMORY_BLOCK
         cerr << "MemoryBlock " << this
              << "   Using preexisting data : " << data_ << endl;
#endif
      }

      virtual ~MemoryBlock()
      {
#ifdef LTL_DEBUG_MEMORY_BLOCK
         cerr << "MemoryBlock " << this
              << "   Deleting : " << data_ << endl;
#endif
         if( data_ )
            delete [] data_;
         data_ = NULL;
      }

      value_type* data()
      {
         addReference();
         return data_;
      }

      value_type* data() const
      {
         addReference();
         return data_;
      }

      void addReference()
      {
         References_.IncAndFetch();
      }

      void removeReference()
      {
         if( References_.DecAndFetch() == 0 )
            delete this;
      }

      int references() const
      {
         return References_.Counter();
      }

      void describeSelf() const
      {
         cout << "  MemBlock at         : " << this << endl;
         cout << "     References       : " << References_.Counter() << endl;
         cout << "     Data ptr         : " << data_ << endl;
      }

   protected:   // Data members
      value_type*   restrict_ data_;
      SynchronizedCounter References_;
};


/*!
  Provides a reference-counted block of mapped memory. This block
  may be referred to by multiple array objects. The memory
  is automatically deallocated when the last referring object is destructed.
*/
template<class T>
class MemoryMapBlock : public MemoryBlock<T>
{

   public:

      MemoryMapBlock( const int items, const char * filename = NULL )
         : MemoryBlock<T>(),
         length_(items * sizeof(T)), savemmap_(filename != NULL)
      {
         if(savemmap_)
            strcpy(filename_, filename);
         else
            strcpy(filename_, "./ltl.XXXXXX");
         const int tmp_fd = savemmap_ ?
            open(filename_, O_RDWR|O_CREAT,
                 S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH)
            : mkstemp(filename_);
         if(tmp_fd < 0)
            throw RangeException("Cannot open memory map file.");
         // set length of file
         if(length_ > 0)
         {
            if( ftruncate(tmp_fd, length_) < 0)
               throw RangeException("Cannot set file size for map file");
            if(fsync(tmp_fd) < 0)
               throw RangeException("Cannot sync map file");
            // map file
            this->data_ = (T *) mmap(NULL, length_,
                                     PROT_READ|PROT_WRITE, MAP_SHARED,
                                     tmp_fd, 0);
            if((unsigned char *)(this->data_) == (unsigned char *) MAP_FAILED)
               throw RangeException("Cannot map mapfile");
         }         
         close(tmp_fd);
      }

      virtual ~MemoryMapBlock()
      {
         if( this->data_ )
         {
            munmap((char *)(this->data_), length_);
            this->data_ = NULL;
            if(!savemmap_)
               remove(filename_);
         }
      }

   private:   // Data members
      const size_t  length_;
      bool          savemmap_;
      char          filename_[16];      
};

}

#endif // __LTL_MEMBLOCK__
