#ifndef CHECK_DEVICE_CUH
#define CHECK_DEVICE_CUH

#include "cuda_utils_check.hpp"
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>
int CHECK_Device(char **argv) {

  printf("%s Starting...\n", argv[0]);

  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount == 0) {
    printf("There are no available device(s) that support CUDA\n");
  } else {
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
  }

  // chapter 策略
  //  int dev = 0;
  //  CHECK_Runtime(cudaSetDevice(
  //      dev)); //
  //      将当前线程的使用设备设置为ID为dev的GPU设备，dev表示要使用的GPU设备的ID，设备ID从0开始，依次对应于系统中可用的GPU设备
  //  cudaDeviceProp deviceProp;
  //  CHECK_Runtime(cudaGetDeviceProperties(&deviceProp, dev));
  //  printf("Device name is: \"%s\"\n", dev, deviceProp.name);

  // 改用leimao策略
  int device_id{0};
  cudaGetDevice(&device_id);
  cudaDeviceProp device_prop;
  CHECK_Runtime(cudaGetDeviceProperties(&device_prop, device_id));
  std::cout << "Device Name: " << device_prop.name << std::endl;
  return EXIT_SUCCESS;
}

#endif