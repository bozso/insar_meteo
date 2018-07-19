module main_functions
implicit none
character(1), parameter :: nl = char(10)

contains
    subroutine hello()
        print *, "Hello World!"
    end subroutine hello

end module main_functions
