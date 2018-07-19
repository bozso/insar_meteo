module utils
use iso_fortran_env, only: erru=>error_unit
implicit none

private
public sp, dp, erru, nl

integer, parameter :: sp = selected_real_kind(6, 37)
integer, parameter :: dp = selected_real_kind(15, 307)
character(1), parameter :: nl = char(10)

end module utils
