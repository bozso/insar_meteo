from subprocess import check_output, CalledProcessError, STDOUT
from shlex import split
from os.path import basename

flags = ""

inputfile = "daisy.c"

def main():

    cmd = "gcc {} -o {}".format(inputfile, basename(inputfile).split('.')[0])
    
    try:
        cmd_out = check_output(split(cmd + " " + flags), stderr=STDOUT)
    except CalledProcessError as e:
        print("Compilation failed, compiler command: '{}'".format(cmd))
        print("OUTPUT OF THE COMMAND: \n{}".format(e.output.decode()))

if __name__ == "__main__":
    main()
