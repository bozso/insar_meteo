#ifndef ERROR_HH
#define ERROR_HH

enum status {
    err_ok = 0,
    err_some
}

bool err_test_and_clear(int& err);
void err_set(status const& err);
void err_clear(void);
status const err_get(void);

#endif
