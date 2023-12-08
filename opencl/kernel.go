package opencl

// #include "opencl.h"
import "C"
import (
	"fmt"
	"unsafe"
)

type Kernel struct {
	kernel C.cl_kernel
}

func createKernel(program Program, kernelName string) (*Kernel, error) {
	kn := C.CString(kernelName)
	defer C.free(unsafe.Pointer(kn))

	var errInt clError
	kernel := C.clCreateKernel(program.program, kn, (*C.cl_int)(&errInt))
	if errInt != clSuccess {
		fmt.Println("Error code", errInt)
		return nil, clErrorToError(errInt)
	}

	return &Kernel{kernel}, nil
}

func (k Kernel) SetArg(argIndex uint32, argSize uint64, argValue interface{}) error {
	var argPtr unsafe.Pointer
	switch argValue.(type) {
	case *Buffer:
		argPtr = unsafe.Pointer(argValue.(*Buffer))
	case *byte:
		argPtr = unsafe.Pointer(argValue.(*byte))
	case *float32:
		argPtr = unsafe.Pointer(argValue.(*float32))
	case *float64:
		argPtr = unsafe.Pointer(argValue.(*float64))
	case *uint32:
		argPtr = unsafe.Pointer(argValue.(*uint32))
	case *uint64:
		argPtr = unsafe.Pointer(argValue.(*uint64))
	default:
		return fmt.Errorf("Unknown type for argValue: %T", argValue)
	}

	errInt := clError(C.clSetKernelArg(
		k.kernel,
		C.cl_uint(argIndex),
		C.size_t(argSize),
		argPtr,
	))
	return clErrorToError(errInt)
}

func (k Kernel) Release() {
	C.clReleaseKernel(k.kernel)
}
