package opencl

// #include "opencl.h"
import "C"
import (
	"fmt"
	"unsafe"
)

type CommandQueue struct {
	commandQueue C.cl_command_queue
}

func createCommandQueue(context Context, device Device) (*CommandQueue, error) {
	var errInt clError
	queue := C.clCreateCommandQueue(
		context.context,
		device.deviceID,
		0,
		(*C.cl_int)(&errInt),
	)
	if errInt != clSuccess {
		return nil, clErrorToError(errInt)
	}

	return &CommandQueue{queue}, nil
}

func (c CommandQueue) EnqueueNDRangeKernel(kernel *Kernel, workDim uint32, globalWorkSize uint64) error {
	errInt := clError(C.clEnqueueNDRangeKernel(c.commandQueue,
		kernel.kernel,
		C.cl_uint(workDim),
		nil,
		(*C.size_t)(&globalWorkSize),
		nil, 0, nil, nil))
	return clErrorToError(errInt)
}

func (c CommandQueue) EnqueueReadBuffer(buffer *Buffer, blockingRead bool, dataLen uint64, dataPtr interface{}) error {
	var br C.cl_bool
	if blockingRead {
		br = C.CL_TRUE
	} else {
		br = C.CL_FALSE
	}

	var ptr unsafe.Pointer

	switch t := dataPtr.(type) {
	case []float32:
		ptr = unsafe.Pointer(&t[0])
	case []float64:
		ptr = unsafe.Pointer(&t[0])
	case []uint32:
		ptr = unsafe.Pointer(&t[0])
	case []uint64:
		ptr = unsafe.Pointer(&t[0])
	case []byte:
		ptr = unsafe.Pointer(&t[0])
	default:
		return fmt.Errorf("data type [%T] is not supported", t)
	}

	errInt := clError(C.clEnqueueReadBuffer(c.commandQueue,
		buffer.buffer,
		br,
		0,
		C.size_t(dataLen),
		ptr,
		0, nil, nil))
	return clErrorToError(errInt)
}

func (c CommandQueue) EnqueueWriteBuffer(buffer *Buffer, blockingWrite bool, offset uint64, dataLen uint64, dataPtr unsafe.Pointer) error {
	var br C.cl_bool
	if blockingWrite {
		br = C.CL_TRUE
	} else {
		br = C.CL_FALSE
	}

	errInt := clError(C.clEnqueueWriteBuffer(c.commandQueue,
		buffer.buffer,
		br,
		C.size_t(offset),
		C.size_t(dataLen),
		dataPtr,
		0, nil, nil))
	return clErrorToError(errInt)
}

func (c CommandQueue) Release() {
	C.clReleaseCommandQueue(c.commandQueue)
}

func (c CommandQueue) Flush() {
	C.clFlush(c.commandQueue)
}

func (c CommandQueue) Finish() {
	C.clFinish(c.commandQueue)
}
