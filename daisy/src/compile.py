from subprocess import check_output, CalledProcessError, STDOUT
from shlex import split
from os.path import basename

def cmd(command, ret_out=True):
    try:
        cmd_out = check_output(split(command), stderr=STDOUT)
    except CalledProcessError as e:
        print("Command failed, command: '{}'".format(cmd))
        print("OUTPUT OF THE COMMAND: \n{}".format(e.output.decode()))
        print("RETURNCODE: \n{}".format(e.returncode))
    
    if ret_out:
        return cmd_out.decode()


CC = "gcc"
flags = "-std=c99 -static " + cmd("gsl-config --cflags --libs")
inputfile = "daisy.c"

def main():

    comp_cmd = "{} {} -o {}".format(CC, inputfile,
                                    basename(inputfile).split('.')[0])
    comp_cmd += " " + flags
    
    try:
        cmd_out = check_output(split(comp_cmd), stderr=STDOUT)
    except CalledProcessError as e:
        print("Compilation failed, compiler command: '{}'".format(comp_cmd))
        print("OUTPUT OF THE COMMAND: \n{}".format(e.output.decode()))

if __name__ == "__main__":
    main()
