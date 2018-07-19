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

program inmet
    use utils, only: dp
    
    real(dp), allocatable :: a(:,:), b(:,:)
    
    allocate(a(25000,25000))
    allocate(b(25000,25000))
    
    do ii = 1, 25000
        do jj = 1, 25000
            a(jj,ii) = real(jj + ii, dp)
            b(jj,ii) = real(ii + jj, dp)
        end do
    end do
    
    !write(erru, *) "Error! Asd!"
    !stop 9
end program inmet
