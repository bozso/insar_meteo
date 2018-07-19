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

module fitting
contains
    function polyfit(vx, vy, d)
    ! modified from Rosetta Code
        implicit none

        integer, intent(in)                 :: d
        real*8, dimension(d+1)              :: polyfit
        real*8, dimension(:), intent(in)    :: vx, vy

        real*8, dimension(:,:), allocatable :: X
        real*8, dimension(:,:), allocatable :: XT
        real*8, dimension(:,:), allocatable :: XTX, C

        integer :: i, j

        integer :: n

        n = d + 1

        allocate(XT(n, size(vx)))
        allocate(X(size(vx), n))
        allocate(XTX(n, n))
        allocate(C(n, n))

        ! prepare the matrix
        do i = 0, d
            do j = 1, size(vx)
                X(j, i+1) = vx(j)**i
            end do
        end do

        XT  = transpose(X)
        XTX = matmul(XT, X)

        call inverse(XTX, C, n)

        polyfit = matmul( matmul(C, XT), vy)

        deallocate(X)
        deallocate(XT)
        deallocate(XTX)

    end function polyfit

    function eval_poly(poly, x)
    ! Evaluates a polynom defined by poly at x positions. Poly
    ! should contain the polynom coefficients from lowest to highest
    ! order. Evaluation is carried out with Horner's method.

        implicit none

        integer :: n_poly, n_x, i, j
        real*8 :: poly(:), x(:)
        real*8, allocatable :: eval_poly(:)

        n_poly = size(poly)
        n_x = size(x)

        allocate(eval_poly(n_x))

        do j = 1, n_x
            eval_poly(j) = poly(n_poly) * x(j)
            do i = n_poly - 1, 2
                eval_poly(j) = (eval_poly(j) + poly(i)) * x(j)
            end do
            eval_poly(j) = eval_poly(j) + poly(1)
        end do
    end function eval_poly

  subroutine inverse(a,c,n)
!============================================================
! Inverse matrix
! Method: Based on Doolittle LU factorization for Ax=b
! Alex G. December 2009
!-----------------------------------------------------------
! input ...
! a(n,n) - array of coefficients for matrix A
! n      - dimension
! output ...
! c(n,n) - inverse matrix of A
! comments ...
! the original matrix a(n,n) will be destroyed
! during the calculation
!===========================================================
implicit none
integer n
double precision a(n,n), c(n,n)
double precision L(n,n), U(n,n), b(n), d(n), x(n)
double precision coeff
integer i, j, k

! step 0: initialization for matrices L and U and b
! Fortran 90/95 aloows such operations on matrices
L=0.0
U=0.0
b=0.0

! step 1: forward elimination
do k=1, n-1
   do i=k+1,n
      coeff=a(i,k)/a(k,k)
      L(i,k) = coeff
      do j=k+1,n
         a(i,j) = a(i,j)-coeff*a(k,j)
      end do
   end do
end do

! Step 2: prepare L and U matrices
! L matrix is a matrix of the elimination coefficient
! + the diagonal elements are 1.0
do i=1,n
  L(i,i) = 1.0
end do
! U matrix is the upper triangular part of A
do j=1,n
  do i=1,j
    U(i,j) = a(i,j)
  end do
end do

! Step 3: compute columns of the inverse matrix C
do k=1,n
  b(k)=1.0
  d(1) = b(1)
! Step 3a: Solve Ld=b using the forward substitution
  do i=2,n
    d(i)=b(i)
    do j=1,i-1
      d(i) = d(i) - L(i,j)*d(j)
    end do
  end do
! Step 3b: Solve Ux=d using the back substitution
  x(n)=d(n)/U(n,n)
  do i = n-1,1,-1
    x(i) = d(i)
    do j=n,i+1,-1
      x(i)=x(i)-U(i,j)*x(j)
    end do
    x(i) = x(i)/u(i,i)
  end do
! Step 3c: fill the solutions x(n) into column k of C
  do i=1,n
    c(i,k) = x(i)
  end do
  b(k)=0.0
end do
end subroutine inverse
end module fitting
