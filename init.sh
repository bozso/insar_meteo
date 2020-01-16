root="$(realpath $1)"

export PYTHONPATH="${PYTHONPATH}:${root}"
export PATH="${PATH}:${root}/bin"
export MATLABPATH="${MATLABPATH}:${root}/Matlab"
