#include "error.hh"

_Thread_local static int error = err_ok;

bool err_test_and_clear(int& err)
{
  // Simplistic code.  See also "C11 7.17 Atomics <stdatomic.h>" 
  if (err) {
    err = error;
  }
  
  error = err_ok;
  return error != err_ok;
}


void err_set(status const& e)
{
    error = e;
}

void err_clear(void)
{
    error = err_ok;
}

status const err_get(void)
{
    return error;
}
