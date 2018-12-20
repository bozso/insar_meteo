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

#define palloc(type, num) (type *) alloc(sizeof(type) * (num))


enum status {
    err_ok = 0,
    err_some
};

#define err_check(ret)             \
do {                               \
    if (PyErr_Occurred() != NULL)  \
        return ret;                \
} while(0)



#if 0
bool err_test_and_clear(int& err);
void err_set(status const& err);
void err_clear(void);
status const err_get(void);
#endif


typedef const double cdouble;

/*******************************
 * WGS-84 ELLIPSOID PARAMETERS *
 *******************************/

// RADIUS OF EARTH
static const double R_earth = 6372000.0;

static const double WA = 6378137.0;
static const double WB = 6356752.3142;

// (WA*WA-WB*WB)/WA/WA
static const double E2 = 6.694380e-03;


static const double pi = 3.14159265358979;
static const double pi_per_4 = pi / 4.0;

static const double deg2rad = 1.745329e-02;
static const double rad2deg = 5.729578e+01;


/**************
 * for macros *
 **************/

#define FOR(ii, max) for(size_t (ii) = (max); (ii)--; )
#define FORZ(ii, max) for(size_t (ii) = 0; (ii) < max; ++(ii))
#define FORS(ii, min, max, step) for(size_t (ii) = (min); (ii) < (max); (ii) += (step))
#define FOR1(ii, min, max) for(size_t (ii) = (min); (ii) < (max); ++(ii))


/****************
 * IO functions *
 ****************/

//void print(char const* fmt, ...);
//void println(char const* fmt, ...);

//void error(char const* fmt, ...);
//void errorln(char const* fmt, ...);
//void perrorln(char const* perror_str, char const* fmt, ...);

#define _log printf("File: %s line: %d\n", __FILE__, __LINE__)


struct File {
    FILE *_file;
    
    File(): _file(NULL) {};
    File(FILE *_file): _file(_file) {};
    ~File();

    bool open(char const* path, char const* mode);
    void close();

    int read(char const* fmt, ...) const;
    int write(char const* fmt, ...) const;
    
    int read(size_t const size, size_t const num, void* var) const;
    int write(size_t const size, size_t const num, void const* var) const;
};


#define str_equal(str1, str2) (not strcmp((str1), (str2)))

#endif // UTILS_HPP
