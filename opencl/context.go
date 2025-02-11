package opencl

import (
	"io/ioutil"
	"unsafe"
)

// #include "opencl.h"
import "C"

type Context struct {
	context C.cl_context
}

func createContext(device Device) (*Context, error) {
	// TODO add more functionality. Super simple context creation right now
	var errInt clError
	ctx := C.clCreateContext(
		nil,
		1,
		(*C.cl_device_id)(&device.deviceID),
		nil,
		nil,
		(*C.cl_int)(&errInt),
	)
	if errInt != clSuccess {
		return nil, clErrorToError(errInt)
	}

	return &Context{ctx}, nil
}

func (c *Context) CreateCommandQueue(device Device) (*CommandQueue, error) {
	return createCommandQueue(c, device)
}

func (c *Context) CreateProgramFromSourceFile(programCodeFileName string) (*Program, error) {
	programCode, err := ioutil.ReadFile(programCodeFileName)
	if err != nil {
		return nil, err
	}

	return createProgramWithSource(c, string(programCode))
}

func (c *Context) CreateProgramWithSource(programCode string) (*Program, error) {
	return createProgramWithSource(c, programCode)
}

func (c *Context) CreateBuffer(memFlags []MemFlags, hostPtr unsafe.Pointer, size uint64) (*Buffer, error) {
	return createBuffer(c, memFlags, hostPtr, size)
}

func (c *Context) Release() {
	C.clReleaseContext(c.context)
}
