package main

import (
	"fmt"
	"unsafe"

	"strings"

	"github.com/MasterDimmy/go-opencl/opencl"
)

const (
	deviceType = opencl.DeviceTypeAll

	dataSize = 128

	programCode = `
kernel void kern(global float* out)
{
	size_t i = get_global_id(0);
	out[i] = i;
}
`
)

func errpanic(err error) {
	if err != nil {
		panic(err.Error())
	}
}

func printHeader(name string) {
	fmt.Println(strings.ToUpper(name))
	for _ = range name {
		fmt.Print("=")
	}
	fmt.Println()
}

func printInfo(platform opencl.Platform, device opencl.Device) {
	var platformName string
	err := platform.GetInfo(opencl.PlatformName, &platformName)
	if err != nil {
		panic(err)
	}

	var vendor string
	err = device.GetInfo(opencl.DeviceVendor, &vendor)
	if err != nil {
		panic(err)
	}

	fmt.Println()
	printHeader("Using")
	fmt.Println("Platform:", platformName)
	fmt.Println("Vendor:  ", vendor)
}

func main() {
	platforms, err := opencl.GetPlatforms()
	if err != nil {
		panic(err)
	}

	printHeader("Platforms")

	foundDevice := false

	var platform opencl.Platform
	var device opencl.Device
	var name string
	for _, curPlatform := range platforms {
		err = curPlatform.GetInfo(opencl.PlatformName, &name)
		if err != nil {
			panic(err)
		}

		var devices []opencl.Device
		devices, err = curPlatform.GetDevices(deviceType)
		if err != nil {
			panic(err)
		}

		// Use the first available device
		if len(devices) > 0 && !foundDevice {
			var available bool
			err = devices[0].GetInfo(opencl.DeviceAvailable, &available)
			if err == nil && available {
				platform = curPlatform
				device = devices[0]
				foundDevice = true
			}
		}

		version := curPlatform.GetVersion()
		fmt.Printf("Name: %v, devices: %v, version: %v\n", name, len(devices), version)
	}

	if !foundDevice {
		panic("No device found")
	}

	printInfo(platform, device)

	context, err := device.CreateContext()
	errpanic(err)
	defer context.Release()

	if context == nil {
		panic("context not created")
	}

	commandQueue, err := context.CreateCommandQueue(device)
	errpanic(err)
	defer commandQueue.Release()
	if commandQueue == nil {
		panic("command queue is empty")
	}

	program, err := context.CreateProgramWithSource(programCode)
	errpanic(err)
	defer program.Release()
	if program == nil {
		panic("program is empty")
	}

	var log string
	err = program.Build(device, "", &log)
	errpanic(err)

	kernel, err := program.CreateKernel("kern")
	errpanic(err)
	defer kernel.Release()
	if kernel == nil {
		panic("kernel is nil")
	}

	// * * * * * test commandQueue buffer read and write * * * * *
	var someData = []uint64{12, 14}
	var someDataSz = uint64(len(someData)) * uint64(unsafe.Sizeof(&someData[0]))
	fmt.Printf("someDatasz: %d\n", someDataSz)

	// create buffer
	bufferTest, err := context.CreateBuffer([]opencl.MemFlags{opencl.MemReadWrite}, nil, someDataSz)
	errpanic(err)
	defer bufferTest.Release()

	// write buffer
	err = commandQueue.EnqueueWriteBuffer(bufferTest, true, someDataSz, unsafe.Pointer(&someData[0]))
	errpanic(err)

	// read written buffer
	var retData = make([]uint64, 2)
	err = commandQueue.EnqueueReadBuffer(bufferTest, true, retData)
	errpanic(err)

	fmt.Printf("read data: %v\n", retData)
	if retData[0] != someData[0] && retData[1] != someData[1] {
		panic("incorrect read")
	}

	// ******************** end test write / read buffer

	// * * * * * test 2 commandQueue buffer read and write * * * * *
	// create buffer
	bufferTest2, err := context.CreateBuffer([]opencl.MemFlags{opencl.MemReadWrite, opencl.MemUseHostPtr, opencl.MemCopyHostPtr}, unsafe.Pointer(&someData[0]), someDataSz)
	errpanic(err)
	defer bufferTest2.Release()

	// read written buffer
	var retData2 = make([]uint64, 2)
	err = commandQueue.EnqueueReadBuffer(bufferTest2, true, retData2)
	errpanic(err)

	fmt.Printf("read data: %v\n", retData2)
	if retData2[0] != someData[0] && retData2[1] != someData[1] {
		panic("incorrect read")
	}

	// ******************** end test 2 write / read buffer

	buffer, err := context.CreateBuffer([]opencl.MemFlags{opencl.MemWriteOnly}, nil, dataSize*4)
	if err != nil {
		panic(err)
	}
	defer buffer.Release()

	err = kernel.SetArg(0, buffer.Size(), buffer)
	if err != nil {
		panic(err)
	}

	err = commandQueue.EnqueueNDRangeKernel(kernel, 1, []uint64{dataSize})
	if err != nil {
		panic(err)
	}

	commandQueue.Flush()
	commandQueue.Finish()

	data := make([]float32, dataSize)

	err = commandQueue.EnqueueReadBuffer(buffer, true, data)
	if err != nil {
		panic(err)
	}

	fmt.Println()
	printHeader("Output")
	for _, item := range data {
		fmt.Printf("%v ", item)
	}
	fmt.Println()
}
