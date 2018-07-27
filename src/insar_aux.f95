module insar_aux

use iso_fortran_env, only: erru=>error_unit
implicit none

integer, parameter :: sp = selected_real_kind(6, 37)
integer, parameter :: dp = selected_real_kind(15, 307)
character(1), parameter :: nl = char(10)

contains
    subroutine test
        print *, "Hello!"
        write(erru, *) "Error!"
        stop
    end subroutine test

end module insar_aux
