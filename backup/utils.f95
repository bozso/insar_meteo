module utils
use iso_fortran_env, only: erru=>error_unit
implicit none

integer, parameter :: dp = selected_real_kind(15, 307)

private
public sp, dp, nl, ell_cart, cart_ell, rad2deg, deg2rad, success

!type orbit_fit
!    integer :: deg, is_centered
!    real(dp) :: t_start, t_stop
!    real(dp) :: coeffs(3,:), mean_coords(3)
!end type orbit_fit

integer, parameter :: sp = selected_real_kind(6, 37)
character(1), parameter :: nl = char(10)

! *******************************
! * WGS-84 ellipsoid parameters *
! *******************************

real(dp), parameter :: R_earth = 6372000.0
real(dp), parameter :: WA = 6378137.0
real(dp), parameter :: WB = 6356752.3142

! (WA^2 - WB^2) / WA^2
real(dp), parameter :: E2 = 6.694380e-03

! ************************
! * degrees, radians, PI *
! ************************

real(dp), parameter :: PI = 3.14159265358979
real(dp), parameter :: deg2rad = 1.745329e-02
real(dp), parameter :: rad2deg = 5.729578e+01

integer, parameter :: success = 0

contains
    pure subroutine ell_cart(lon, lat, height, x, y, z)
        real(dp), intent(in) :: lon, lat, height
        real(dp), intent(out) :: x, y, z
        
        real(dp) :: n
        
        n = WA / sqrt(1.0 - E2 * sin(lat) * sin(lat))
        
        x = (              n + height) * cos(lat) * cos(lon);
        y = (              n + height) * cos(lat) * sin(lon);
        z = ( (1.0 - E2) * n + height) * sin(lat);
    end subroutine

    ! From cartesian to ellipsoidal coordinates.
    pure subroutine cart_ell (x, y, z, lon, lat, height)
        real(dp), intent(out) :: lon, lat, height
        real(dp), intent(in) :: x, y, z
    
        real(dp) :: n, p, o, so, co

        n = (WA * WA - WB * WB)
        p = sqrt(x * x + y * y)

        o = atan(WA / p / WB * z)
        so = sin(o)
        co = cos(o);
        
        o = atan( (z + n / WB * so * so * so) / (p - n / WA * co * co * co) );
        
        so = sin(o)
        co = cos(o);
        
        n= WA * WA / sqrt(WA * co * co * WA + WB * so * so * WB);

        lat = o;
    
        o = atan(y/x);
        if(x < 0.0) o = o + PI
        lon = o;
        height = p / co - n;
    end subroutine cart_ell

!    ! Calculate satellite position based on fitted polynomial orbits at `time`.
!    pure subroutine calc_pos(orb, time, x, y, z)
!        type(orbit_fit), intent(in) :: orb
!        real(dp), intent(in) :: time
!        real(dp), intent(out) :: x, y, z
        
!        integer n_poly, deg, ii
        
!        n_poly = orb%deg + 1
        
!        real(dp) :: coeffs(3, n_poly)
        
!        if(n_poly == 2) then
!            x = coeffs(1,1) + coeffs(1,2) * time;
!            y = coeffs(2,1) + coeffs(2,2) * time;
!            z = coeffs(3,1) + coeffs(3,2) * time;
!        else
!            ! highest degree
!            x = coeffs(1, n_poly) * time;
!            y = coeffs(2, n_poly) * time;
!            z = coeffs(3, n_poly) * time;
            
!            do ii = n_poly - 1, 2, -1
!                x = (x + coeffs(1,ii)) * time;
!                y = (y + coeffs(2,ii)) * time;
!                z = (z + coeffs(3,ii)) * time;
!            end do
            
!            ! lowest degree
!            x = x + coeffs(1,1);
!            y = y + coeffs(2,1);
!            z = z + coeffs(3,1);
!        end if
        
!    if (orb%centered) then
!        x = x + orb%coords_mean(1);
!        y = y + orb%coords_mean(2);
!        z = z + orb%coords_mean(3);
!    end if
    
!    end subroutine calc_pos

end module utils
