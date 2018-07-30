! Copyright (C) 2018  István Bozsó
! 
! This program is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
! 
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
! 
! You should have received a copy of the GNU General Public License
! along with this program.  If not, see <http://www.gnu.org/licenses/>.

include "utils.f95"

module insar_aux
use iso_fortran_env, only: erru=>error_unit
use utils
implicit none

private
public test

contains
!    subroutine azi_inc(t_start, t_stop, coeffs, deg, centered, n)
!        integer, intent(in) :: n
!        real(dp), intent(in) :: t_start, t_stop, coeffs(3,n)
        
        
        
!    end subroutine

    subroutine test
        integer :: ii
        real(dp) :: x, y, z, lon, lat, height

        lon = 45.0 * deg2rad; lat = 25.0 * deg2rad; height = 63900.0;
        
        do ii = 1, 10000000
            call ell_cart(lon, lat, height, x, y, z)
            call cart_ell(x, y, z, lon, lat, height)
        end do
        print *, lon * rad2deg, lat * rad2deg, height
        
    end subroutine test

end module insar_aux
