#define __USE_MINGW_ANSI_STDIO 1
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "cl_head.h"

#ifdef FPGA
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x500000)

static cl_device_id device_id = NULL;
static cl_context context = NULL;
static cl_command_queue command_queue = NULL;
static cl_mem memobjA = NULL;
static cl_mem memobjB = NULL;
static cl_mem memobjC = NULL;
static cl_program program = NULL;
static cl_kernel kernel = NULL;
static cl_kernel nkernel[10];
static cl_platform_id platform_id = NULL;

int gemm_fpga_init () {
  int i,j,k;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret;
  cl_int ret1,ret2,ret3;

    const char *k_name[2]={"gemm_nn9W","gemm_nnfW"};
    find_CnKQ(
        "Intel(R) FPGA SDK for OpenCL(TM)",
        "gemm1.aocx",
        2,
        k_name,
        &context, nkernel, &command_queue
    );
return CL_SUCCESS;

  FILE *fp;
  char fileName[] = "./gemm1.aocx";
  const unsigned char *source_str;
  size_t source_size;

  fprintf(stderr,"gemm_fpga_init_start\n");
/* Load the source code containing the kernel*/
  fp = fopen (fileName, "r");
  if (!fp) {
	fprintf (stderr, "Failed to load kernel.\n");
	exit (1);
  }else printf("fileName=%s\n",fileName);
  source_str = (const unsigned char *) malloc (MAX_SOURCE_SIZE);
  source_size = fread ((void*)source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose (fp);

/* Get Platform and Device Info */
  ret = clGetPlatformIDs (1, &platform_id, &ret_num_platforms);
  if(ret != CL_SUCCESS){
	fprintf(stderr,"Faild clGetPlatform %d\n",ret);
	exit(ret);
  }
  cl_ulong local_mem;
  char platform_name[1024], device_name[1024];
  clGetPlatformInfo(platform_id,CL_PLATFORM_NAME,sizeof(platform_name),platform_name,NULL);
  printf("\tNo.%d-\"%s\"\n",i,platform_name);
  ret = clGetDeviceIDs (platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
  printf("\t%d devices\n",ret_num_devices);
  for(j=0;j<ret_num_devices;j++){
    clGetDeviceInfo(device_id, CL_DEVICE_NAME,sizeof(device_name),device_name,NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong),&local_mem,NULL);
    printf("\t\tNo.%d-\"%s\" : LOCAL_MEM_SIZE=%lu\n",j,device_name,local_mem);
  }
  ret =
	clGetDeviceIDs (platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id,
					&ret_num_devices);
  if(ret != CL_SUCCESS){
	fprintf(stderr,"Faild clGetDeviceIDs %d\n",ret);
	exit(ret);
  }
  clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong),&local_mem,NULL);
  fprintf(stderr,"LOCAL_MEM_SIZE=%lu\n",local_mem);

/* Create OpenCL context */
  context = clCreateContext (NULL, 1, &device_id, NULL, NULL, &ret);
  if(ret != CL_SUCCESS){
	fprintf(stderr,"Faild clCreateContext %d\n",ret);
	exit(ret);
  }

/* Create Command Queue */
  command_queue = clCreateCommandQueue (context, device_id, 0, &ret);
  if(ret != CL_SUCCESS){
	fprintf(stderr,"Faild clCreateCommandQueue %d\n",ret);
	exit(ret);
  }

/* Create Kernel Program from the source */
  fprintf(stderr,"load from source onX86\n");
  program =
	clCreateProgramWithSource (context, 1, (const char **) &source_str,
							   (const size_t *) &source_size, &ret);
  if(ret != CL_SUCCESS){
	fprintf(stderr,"Faild clCreateProgramWithXXXX %d\n",ret);
	exit(ret);
  }

/* Build Kernel Program */
  ret = clBuildProgram (program, 1, &device_id, NULL, NULL, NULL);
  if(ret != CL_SUCCESS){
	fprintf(stderr,"Faild clBuildProgram %d\n",ret);
	exit(ret);
  }

  cl_kernel kernels[10];
  cl_uint n_kernels=1;
  ret = clCreateKernelsInProgram(program,2,kernels,&n_kernels);
  fprintf(stderr,"In Program kernels = %d\n",n_kernels);
  if(ret != CL_SUCCESS){
	fprintf(stderr,"Faild clCreateKernelsInProgram %d\n",ret);
//	exit(ret);
  }else{

	  char name[64];
	  size_t info_size;
	  for(i=0;i<n_kernels;i++){
		  ret = clGetKernelInfo(kernels[i],CL_KERNEL_FUNCTION_NAME,32,name,&info_size);
		  fprintf(stderr,"In Program kernel[%d] name = %s\n",i,name);
		  clReleaseKernel(kernels[i]);
	  }
  }

/* Create OpenCL Kernel */
  char kernel_name[128]="gemm_nn_20W";
  kernel = clCreateKernel (program, kernel_name, &ret);
  if(ret != CL_SUCCESS){
	fprintf(stderr,"Faild clCreateKernel %d\n",ret);
	exit(ret);
  }else{fprintf(stderr,"Initialization completed kernel=%s\n",kernel_name);}
  return ret;
}

void gemm_nn_fpga(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc) {
  cl_int ret,ret1,ret2,ret3;
  memobjA = clCreateBuffer (context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,
					M * K * sizeof (float), A, &ret1);
  memobjB = clCreateBuffer (context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,
					K * N * sizeof (float), B, &ret2);
  memobjC = clCreateBuffer (context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
					M * N * sizeof (float), C, &ret3);
  if(ret1 != CL_SUCCESS || ret2 != CL_SUCCESS || ret3 != CL_SUCCESS){
	fprintf(stderr,"Faild clCreateBuffer %d %d %d\n",ret1,ret2,ret3);
	exit(ret3);
  }
  if(!(K%27)){
    kernel = nkernel[0];
    printf("9W-kernel M/N/K=%d/%d/%d\n",M,N,K);
  }else{
    kernel = nkernel[1];
    printf("fW-kernel M/N/K=%d/%d/%d\n",M,N,K);
  }
/* Set OpenCL Kernel Parameters */
  ret|= clSetKernelArg (kernel, 0, sizeof (cl_int),  &M);               checkErr(ret,"clSetKernelArg-0");
  ret|= clSetKernelArg (kernel, 1, sizeof (cl_int),  &N);               checkErr(ret,"clSetKernelArg-1");
  ret|= clSetKernelArg (kernel, 2, sizeof (cl_int),  &K);               checkErr(ret,"clSetKernelArg-2");
  ret|= clSetKernelArg (kernel, 3, sizeof (cl_float),&ALPHA);           checkErr(ret,"clSetKernelArg-3");
  ret|= clSetKernelArg (kernel, 4, sizeof (cl_mem), (void *) &memobjA); checkErr(ret,"clSetKernelArg-4");
  ret|= clSetKernelArg (kernel, 5, sizeof (cl_int),  &K);               checkErr(ret,"clSetKernelArg-5");
  ret|= clSetKernelArg (kernel, 6, sizeof (cl_mem), (void *) &memobjB); checkErr(ret,"clSetKernelArg-6");
  ret|= clSetKernelArg (kernel, 7, sizeof (cl_int),  &N);               checkErr(ret,"clSetKernelArg-7");
  ret|= clSetKernelArg (kernel, 8, sizeof (cl_mem), (void *) &memobjC); checkErr(ret,"clSetKernelArg-8");
  ret|= clSetKernelArg (kernel, 9, sizeof (cl_int),  &N);               checkErr(ret,"clSetKernelArg-9");

/* Execute OpenCL Kernel */
  ret = clEnqueueTask (command_queue, kernel, 0, NULL, NULL);
  clFinish(command_queue);
  if(ret == CL_SUCCESS){
	  ret = clReleaseMemObject (memobjA);
	  ret = clReleaseMemObject (memobjB);
	  ret = clReleaseMemObject (memobjC);
  }else{fprintf(stderr,"clEnqueueTask Error %d\n",ret);exit(-1);}
  return;
}

void gemm_fpga_finalize(){
  cl_int ret;
/* Finalization */
  ret = clFlush (command_queue);
  ret = clFinish (command_queue);
  ret = clReleaseKernel (kernel);
  ret = clReleaseProgram (program);
  ret = clReleaseCommandQueue (command_queue);
  ret = clReleaseContext (context);

  //free ((void*)source_str);

  if(ret==CL_SUCCESS)fprintf(stderr,"gemm fpga finalized.\n");
  return ;
}
#endif

