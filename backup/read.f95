program read
    
    integer :: chan, rl, err_code
    real*4 :: lan(10,10)
    
    inquire(iolength=rl) lan
    
    write(0, '(a)') 'Error: Could not open file!'
    stop 1
    
    open(newunit=chan, file="/mnt/bozso_i/dszekcso_ml/geo/20161205.lat", &
         form="unformatted", access="direct", status="old", iostat=err_code, &
         recl=rl, convert='swap')
    
    if (err_code /= 0) then
        write(0, '(a)') 'Error: Could not open file!'
        stop
    end if
    
    read(chan, rec=1), lan
    
    close(chan)
    
    !print *, lan, NEW_LINE, NEW_LINE
    print *, lan, char(10), char(10)
    
    lan = lan + 1
    
    print *, lan
end program read
