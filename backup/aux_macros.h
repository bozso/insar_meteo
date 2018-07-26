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

#define OR ||
#define AND &&

/**************************
 * for macros             *
 * REQUIRES C99 standard! *
 **************************/

#define FOR(ii, min, max) for(uint (ii) = (min); (ii) < (max); (ii)++)
#define FORS(ii, min, max, step) for(uint (ii) = (min); (ii) < (max); (ii) += (step))

/**********************************************
 * Documentation and argument number checking *
 **********************************************/

#define print_doc(fun_name) printf("%s", fun_name ## __doc__)

#define aux_checkarg(num, doc)\
({\
    if (argc != ((num) + min_arg)) {\
        errorln("\n Required number of arguments is %d, current number of arguments: %d!",\
                 (num), argc - min_arg);\
        printf((doc));\
        error = err_arg;\
        goto fail;\
    }\
})

/*****************************
 * Switching between modules *
 *****************************/

#define str_isequal(string1, string2) (strcmp((string1), (string2)) == 0)
#define module_select(string) str_isequal(argv[1], string)

/*********************
 * Allocation macros *
 *********************/

#define aux_malloc(ptr, type, num)\
({\
    if (NULL == ((ptr) = (type *) malloc(sizeof(type) * (num))))\
    {\
        fprintf(stderr, "FILE: %s, LINE: %d :: Malloc of %s failed",\
                __FILE__, __LINE__, #ptr);\
        error = err_alloc;\
        goto fail;\
    }\
})

#define aux_free(pointer)\
({\
    if ((pointer) != NULL)\
    {\
        free((pointer));\
        pointer = NULL;\
    }\
})

#define aux_alloc(type, num) (type *) calloc((num) * sizeof(type))	
#define aux_realloc(pointer, type, size) (pointer) = (type *) \
                                     realloc(pointer, (size) * sizeof(type))	

/**************************
 * GSL convenience macros *
 **************************/

#define Mset(mtx, ii, jj, data) gsl_matrix_set((mtx), (ii), (jj), (data))
#define Vset(vector, ii, data) gsl_vector_set((vector), (ii), (data))

#define Mget(mtx, ii, jj) gsl_matrix_get((mtx), (ii), (jj))
#define Vget(vector, ii) gsl_vector_get((vector), (ii))

#define Mptr(mtx, ii, jj) gsl_matrix_ptr((mtx), (ii), (jj))
#define Vptr(vector, ii) gsl_vector_ptr((vector), (ii))

/*************
 * IO macros *
 *************/

#define error(string) fprintf(stderr, string)
#define errorln(format, ...) fprintf(stderr, format "\n", __VA_ARGS__)

#define println(format, ...) printf(format "\n", __VA_ARGS__)

#define Log printf("%s\t%d\n", __FILE__, __LINE__)

#define aux_open(file, path, mode)\
({\
    if (((file) = fopen((path), (mode))) == NULL)\
    {\
        errorln("FILE: %s, LINE: %d :: Failed to open file %s!", __FILE__, __LINE__, path);\
        perror("Error");\
        error = errno;\
        goto fail;\
    }\
})

#define aux_close(file)\
({\
    if ((file) != NULL) {\
        fclose((file));\
        (file) = NULL;\
    }\
})


#define aux_fscanf(file, format, ...)\
({\
    if (fscanf(file, format, __VA_ARGS__) <= 0) {\
        error = err_io;\
        goto fail;\
    }\
})
