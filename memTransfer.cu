#include<cuda_runtime.h>
#include<stdio.h>

int main(int argc, char **argv){
    // set up device
    int dev = 0;
    cudaSetDevice(dev);

    // memory size
    unsigned int isize = 1 << 22;
    unsigned int nbytes = isize * sizeof(float); 

    // get device information
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s starting at", argv[0]); 
    printf("device %d: %s memory size %d nbyte %5.2fMB\n",dev,
        deviceProp.name, isize, nbytes/(1024.0f * 1024.0f));
    
    // allocate the host memory
    float *h_a = (float *)malloc(nbytes);

    // allocate the device memory
    float *d_a;
    cudaMalloc((float **)&d_a, nbytes);

    // initialize the host memory
    memset(h_a, 0.5f, nbytes);

    // transfer data from the host to the device
    cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice);

    // free memory
    cudaFree(d_a);
    free(h_a);

    // reset device
    cudaDeviceReset();
    return EXIT_SUCCESS;

}