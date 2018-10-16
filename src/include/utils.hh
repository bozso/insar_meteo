/* Copyright (C) 2018  István Bozsó
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <string.h>

typedef unsigned int uint;
typedef const unsigned int cuint;
typedef const double cdouble;

#define OK 0
#ifndef EIO
#define EIO 1
#endif
#define EARG 2

#define min_arg 2

/*******************************
 * WGS-84 ELLIPSOID PARAMETERS *
 *******************************/

// RADIUS OF EARTH
#define R_earth 6372000

#define WA 6378137.0
#define WB 6356752.3142

// (WA*WA-WB*WB)/WA/WA
#define E2 6.694380e-03


#ifndef M_PI
#define M_PI 3.14159265358979
#endif

#define DEG2RAD 1.745329e-02
#define RAD2DEG 5.729578e+01

/**************
 * for macros *
 **************/

#define FOR(ii, min, max) for(uint (ii) = (min); (ii) < (max); ++(ii))
#define FORS(ii, min, max, step) for(uint (ii) = (min); (ii) < (max); (ii) += (step))


/****************
 * IO functions *
 ****************/

void error(const char * fmt, ...);
void errorln(const char * fmt, ...);
void perrorln(const char * perror_str, const char * fmt, ...);

void println(const char * fmt, ...);

#define _log println("File: %s line: %d", __FILE__, __LINE__)


struct File {
    FILE *_file;
    
    File(): _file(NULL) {};
    ~File();
};


bool open(File& file, const char * path, const char * mode);
void close(File& file);

int read(const File& file, const char * fmt, ...);
int write(const File& file, const char * fmt, ...);

int read(const File& file, const size_t size, const size_t num, void *var);
int write(const File& file, const size_t size, const size_t num, const void *var);


#define str_equal(str1, str2) (not strcmp((str1), (str2)))

#define ut_check(condition, statement)\
do {\
    if ((condition))\
        statement;\
} while (0)

#endif // UTILS_HPP
