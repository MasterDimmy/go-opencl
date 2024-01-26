#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#define CL_TARGET_OPENCL_VERSION 300

#if __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif