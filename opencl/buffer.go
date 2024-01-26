package opencl

import (
	"fmt"
	"unsafe"
)

// #include "opencl.h"
import "C"

type MemFlags uint64

const (
	MemReadWrite MemFlags = C.CL_MEM_READ_WRITE
	MemWriteOnly          = C.CL_MEM_WRITE_ONLY
	MemReadOnly           = C.CL_MEM_READ_ONLY

	MemUseHostPtr    = C.CL_MEM_USE_HOST_PTR
	MemAllocHostPtr  = C.CL_MEM_ALLOC_HOST_PTR
	MemCopyHostPtr   = C.CL_MEM_COPY_HOST_PTR
	MemHostWriteOnly = C.CL_MEM_HOST_WRITE_ONLY
	MemHostReadOnly  = C.CL_MEM_HOST_READ_ONLY
	MemHostNoAccess  = C.CL_MEM_HOST_NO_ACCESS
)

type Buffer struct {
	buffer C.cl_mem
}

func createBuffer(context *Context, flags []MemFlags, hostPtr unsafe.Pointer, size uint64) (*Buffer, error) {
	flagBitField := uint64(0)
	for _, flag := range flags {
		flagBitField |= uint64(flag)
	}

	var ret Buffer

	var errInt clError
	ret.buffer = C.clCreateBuffer(
		context.context,
		C.cl_mem_flags(flagBitField),
		C.size_t(size),
		hostPtr,
		(*C.cl_int)(&errInt),
	)
	if errInt != clSuccess {
		return nil, clErrorToError(errInt)
	}

	return &ret, nil
}

func (b *Buffer) Size() uint64 {
	return uint64(C.sizeof_cl_mem)
}

func (b *Buffer) Release() error {
	err := C.clReleaseMemObject(b.buffer)
	if err != C.CL_SUCCESS {
		return fmt.Errorf("failed to release memory object: %d", err)
	}
	return nil
}
