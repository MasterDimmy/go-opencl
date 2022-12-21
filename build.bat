set CGO_ENABLED=1
set LD_LIBRARY_PATH=.\opencl\external\lib

set CGO_LDFLAGS="-L.\opencl\external\lib"

go build 

