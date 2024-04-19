# CUDA C 编程权威指南

## 基于CUDA的异构并行计算

硬件提供支持多线程或多进程的平台

对计算机体系结构了解，编写正确且高效的进程



并行性

任务并行：多任务可以独立、大规模的并行执行。重点利用多核系统对任务进行分配

数据并行：处理器可以同时处理很多数据。利用多核系统对数据进行分配。

​	数据划分

​		块划分：每个线程只处理数据的一部分，通常数据具有相同大小

​		周期划分：每个线程作用于数据的多部分

![image-20240414134533312](/home/humble/.config/Typora/typora-user-images/image-20240414134533312.png)

计算机架构

指令流和数据流分类

​	SISD：串行架构

​	SIMD：所有核心只有一个指令流处理不同的数据流（向量机）

​	MISD：每个核心使用多个指令流处理同一个数据流

​	MIMD：多个核心通过多个指令流来处理多个数据流

​	设计目标：降低延迟、提高带宽、提高吞吐量



内存组织方式分类

​	分布式内存的多节点系统

​		处理器有自己的本地内存，处理器之间通过网络进行通信

​	共享内存的多处理器系统

​		多个处理器与同一个物理内存相关联或公用一个低延迟的链路



异构架构

​	异构计算节点包含两个多核CPU插槽和多个众核GPU，GPU通过总线与基于CPU的主机相连。CPU所在位置被称为主机端，GPU称为设备端。异构平台由CPU初始化，CPU负责管理设备端的环境、代码和数据。

![image-20240414140834556](/home/humble/.config/Typora/typora-user-images/image-20240414140834556.png)



面实题：

CPU和GPU分别适合于什么情况

如果一个问题具有较小数据规模、复杂的控制逻辑和或多或少的并行性，最好选择CPU处理，因为CPU有处理复杂逻辑和指令级并行性的能力。包含大量数据并表现出大量的数据并行性，使用GPU。GPU有大量可编程的核心，支持大规模多线程运算，相比CPU有较大的峰值带宽。

CPU线程与GPU线程

CPU线程通常是重量级的实体，操作系统需要切换线程，上下文切换开销大；

GPU线程是高度轻量级，核用来处理大量并发的、轻量级线程，提高吞吐量。



CUDA

![image-20240414142754288](/home/humble/.config/Typora/typora-user-images/image-20240414142754288.png)

CUDA程序：CPU上运行的主机代码和GPU上运行的设备代码

![image-20240414143031229](/home/humble/.config/Typora/typora-user-images/image-20240414143031229.png)

CUDA编程结构

1.分配GPU内存

2.从CPU内存拷贝数据到GPU内存

3.调用CUDA内核函数完成程序指定的计算

4.将数据从GPU移到CPU

5.释放CPU内存空间

~~~c
#include<stdio.h>
__global__ void helloFromGPU(void){
    printf("Hello World from GPU!\n");
}
int main(int argc, char** argv){
    // hello from cpu
    printf("Hello World from CPU!\n");
    
    helloFromGPU<<<1, 10>>>();
    // 显示释放和清空当前进程中与当前设备有关的所有资源
    cudaDeviceReset();  				 
    return 0;
}
~~~

CUDA核中的3个关键抽象：线程组的层次结构，内存的层次结构以及障碍同步



## CUDA编程模型

目标

​	主机和设备分配内存空间

​	在CPU和GPU之间拷贝共享内存

### 内存管理

​		用于执行GPU内存分配的函数：向设备分配一定字节的线性内存，并以devPtr的形式返回指向所分配内存的指针

![image-20240414185146569](/home/humble/.config/Typora/typora-user-images/image-20240414185146569.png)

~~~c
cudaMalloc  //与C语言malloc函数几乎一样

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
	// kind : 1.cudaMemcpyHostToHost 
	// 		  2.cudaMemcpyHostToDevice
	//		  3.cudaMemcpyDeviceToHost
	//		  4.cudaMemcpyDeviceToDevice
	//CPU分配成功：cudaSuccess   否则：cudaErrorMemcpyAllocation
	//CudaMemcpy的调用会导致主机运行阻塞
char* cudaGetErrorString(cudaError_t error);//将错误代码转化为可读信息
~~~



线程层次结构：线程块（block）和 线程块网格（grid）

![image-20240414195237456](/home/humble/.config/Typora/typora-user-images/image-20240414195237456.png)

​	由一个内核启动所产生的所有线程统称为一个网格；同一网格中所有线程共享相同的全局内存空间。一个线程块包含一组线程，统一线程块内的线程可以通过**同步和共享内存**协作。不同块内线程不能协作。



​	blcokIdx(线程块在线程格内的索引)

​	threadIdx(块内线程的索引)

​	坐标基于uint3,可以通过x,y,z三个字段来指定。

​	

​	blockDim(线程块的维度，每个线程块中的线程数量)

​	gridDim（线程格的维度，每个线程网格中的线程块数来表示）

​	块大小的限制因素是可利用的计算资源（如寄存器，共享内存）

​	

​	核函数的调用与主机线程是异步的，核函数调用结束后，控制权立刻返回给主极端。可利用 cudaDeviceSynchronize() 强制主机端程序等待所有核函数执行结束

~~~c
cudaError_t cudaDeviceSynchronize(void);
~~~

​	cudaMemcpy函数在主机与设备间拷贝数据时，主机端隐士同步。



### 核函数编写

| 限定符         | 执行   | 调用                                          | 备注             |
| -------------- | ------ | --------------------------------------------- | ---------------- |
| ____global____ | 设备端 | 可以主极端调用，也可以计算能力为3的设备中调用 | 必须void返回类型 |
| ____device____ | 设备端 | 仅设备端调用                                  |                  |
| ____host____   | 主机端 | 仅主极端调用                                  | 可以省略         |

____device____ 和 ____host____可一起使用，这样函数可以同时在主机和设备端进行编译。

cuda核函数的限制：

1.只能访问设备内存

2.必须具有void返回类型

3.不支持可变数量的参数

4.不支持静态变量

5.显示异步行为



### 处理错误

![image-20240414214718038](/home/humble/.config/Typora/typora-user-images/image-20240414214718038.png)

​		CHECK(cudaDeviceSynchronize()) 核函数启动后添加这个检查点会阻塞主极端线程，是该检查点成为全局屏障。



### 给核函数计时

~~~c
#include<sys/time.h>
double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

double iStart = cpuSecond();
kernel_name<<<grid, block>>>(argument list);
cudaDeviceSynchronize();
double iElaps = cpuSecond() - iStart;

//nvprof --help
~~~



### 组织并行线程

​		使用合适的网格和块大小来正确组织线程，可以对内核性能产生很大的影响。

​		在一个矩阵加法函数中，一个线程通常被分配一个数据元素来处理。

​	1.使用块和线程索引从全局内存中访问指定的数据

​			线程和块索引

​			矩阵中给定点的坐标

~~~c
ix = threadIdx.x + blockIdx.x * blockDim.x;
iy = threadIdx.y + blockIdx.y * blockDim.y;
~~~

​			全局线性内存中的偏移量

~~~c
idx = iy * nx + ix;
~~~

![image-20240415131147045](/home/humble/.config/Typora/typora-user-images/image-20240415131147045.png)

~~~c
//set up execution configuration
dim3 block(4, 2);
dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);

__global__ void printThreadIndex(int *A, const int nx, const int ny){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;
    
    printf("thread_id (%d,%d) block_id (%d,%d) coordinate (%d,%d)"
          	"global index %2d ival %2d\n", threadIdx.x, threadIdx.y,
          	blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]);
}
~~~



### 设备管理

~~~c
//查询关于gpu设备的信息
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);

//确定最优GPU
int numDevice = 0;
cudaGetDeviceCount(&numDevice);
if(numDevice > 1){
    int maxMultiprocessors = 0, maxDevice = 0;
    for(int device = 0; device<numDevice;device++){
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device);
        if(maxMultiprocessors < props.multiProcessorCount){
            maxMultiprocessors = props.multiProcessorCount;
            maxDevice = device;
        }
    }
    cudaSetDevice(maxDevice);
}
~~~



nvidia-smi查询GPU信息

nvidia-smi -L  //系统有多少个GPU

nvidia-smi -q -i 0  // 查询0号GPU详细信息

相关参数

MEMORY

UTILIZATION

ECC

POWER

CLOCK

COMPUTE

PIDS

PERFORMANCE

SUPPORTED_CLOCKS

PAGE_RETIREMENT

ACCOUNTING

nvidia-smi -q -i 0 -d UTILIZATION | tail -n 5

使用环境变量CUDA_VISIBLE_DEVICES可以在运行时指定所选的GPU且无须更改应用程序



## CUDA执行模型

### CUDA执行模型概述

GPU架构SM处理器

1.CUDA核心

2.共享内存/一级缓存

3.寄存器文件

4.加载/存储单元

5.特殊功能单元

6.线程束调度器

​		cuda使用单指令多线程（SIMT）架构来管理和执行线程，每32个线程为一组，称为线程束（warp），线程束中所有线程执行相同的指令，每个线程有自己的指令地址计数器和寄存器状态，利用自身的数据执行当前指令。

​		SIMT与SIMD区别：SIMD要求同一个向量中的所有元素要在一个统一的同步组中一起执行，而SIMT允许统一线程束中的多个线程独立执行，

​		SIMT每个线程都有自己的指令地址计数器，寄存器状态和独立的执行路径

![image-20240417203910896](/home/humble/.config/Typora/typora-user-images/image-20240417203910896.png)

16个Load和Store，允许每个时钟周期内有16个线程计算源地址和目的地址

SFU执行固有指令，正弦，余弦，平方和插值等		

一个线程块只能在一个SM上被调度



Kepler架构

1.强化的SM

2.动态并行

3.Hyper-Q技术



### nvvp nvprof使用



### 线程束执行的本质

一个线程块中线程束的数量 = ceil（一个线程块中线程的数量/线程束大小）

线程束分化：CPU拥有复杂的硬件以执行分支预测；GPU中一个线程执行一条指令，那么同一线程束中的线程执行该指令，如果一个线程束中的线程产生分化，线程束将连续执行每一个分支。

~~~c
/* 线程束分化 */
__global__ vlid mathKernel1(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if(tid % 2 == 0){
        a = 100.0f;
    }else{
        b = 200.0f;
    }
    c[tid] = a + b;
}

/* 交叉存取数据，防止线程束分化 */
__global__ void mathKernel2(void){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if((tid / warpSize) % 2 == 0){
        a = 100.0f;
    }else{
        b = 200.0f;
    }
    c[tid] = a + b;
}
/* nvcc -g -G a.cu -o output_file 编译器不利用分支预测优化*/
/* nvprof --metrics branch_effiency ./outputfile 查看分支效率*/
~~~



​	SM处理器每个线程上下文，整个线程束的生存期是保存在芯片内的，从一个上下文切换到另一个执行上下文没有损失

​	同一个SM中线程块和线程束的数量取决于在SM中可用的且内核所需的寄存器的共享内存的数量



同步：

​	cudaDeviceSynchronize函数可以用来阻塞主机应用程序，直到所有的CUDA操作完成

​	__syncthreads在同一线程块中每个线程都必须等待直至该线程块中所有其他线程都已经达到这个同步点。



可扩展性

可实现占用率（achieved_occupancy)：每个周期内活跃线程束的平均数与最大支持线程束的比值

nvprof --metrics achieved_occupancy ./output_file /* 可实现占用率 */

nvprof --metrics gld_throughput ./output_file   /* 内存读取效率 */

nvprof --metrics gld_efficiency ./output_file  /* 全局加载效率  */



### 并行性

增大并行性：一个方法调整blockDim.x



1.领域并行

~~~c
__global__ void reduceInterleaved(int *g_idata, int *odata, int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int *idata = g_idata + blockIdx.x * blockDim.x;
    if(idx >= n) return;
    
    for(int stride = 1; stride < blockDim.x; stride *= 2){
        if((tid%(2 * stride * tid))==0){
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    if(tid==0) g_odata[threadIdx.x] = idata[0];
}
~~~



2.间域并行

~~~c
__global__ void reduceInterleaved(int *g_idata, int * g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x;
    if(idx >= n) return;
    for(int stride = blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride){
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    if(tid==0) g_odata[blockIdx.x]=idata[0];
}
~~~



3.循环展开

~~~c
__global__ void reduceUnrolling (int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;
    if(idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid < stride){
            idata[tid] += idata[tid+stride];
        }
        __syncthreads();
    }
    if(tid==0) g_odata[blockIdx.x] = idata[0];
}
~~~



4.模板函数

~~~c
template<unsigned int iBlockSize>
__global__ void reduceCompleteUnrollWarps8(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;
    if(idx + blockDim.x * 7 < n){
        int a1 = g_idata[idx];
        int a2 = g_idata[idx+blockDim.x];
        int a3 = g_idata[idx+blockDim.x*2];
        int a4 = g_idata[idx+blockDim.x*3];
        int b1 = g_idata[idx+blockDim.x*4];
        int b2 = g_idata[idx+blockDim.x*5];
        int b3 = g_idata[idx+blockDim.x*6];
        int b4 = g_idata[idx+blockDim.x*7];
        g_idata[idx] = a1+a2+a3+a4+b1+b2+b3+b4;
    }
    __syncthreads();
    if(iBlockSize>=1024&&tid<512) idata[tid] += idata[tid+512];
    __syncthreads();
    if(iBlockSize>=512&&tid<256) idata[tid] += idata[tid+256];
    __syncthreads();
    if(iBlockSize>=256&&tid<128) idata[tid] += idata[tid+128];
    __syncthreads();
    if(iBlockSize>=1128&&tid<64) idata[tid] += idata[tid+64];
    __syncthreads();
	if(tid<32){
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }
    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}

~~~

5.动态并行

在内核函数中执行递归调用

~~~c
if(tid == 0){
    gpuRecursiveReduce<<<1, istride>>>(idata,odata, istride);
    // sync all child grids launched in this block
    cudaDeviceSynchronize();
}
//sync at block level again
__syncthreads();
~~~





## 全局内存

CUDA内存模型

![image-20240418001936371](/home/humble/.config/Typora/typora-user-images/image-20240418001936371.png)

1.寄存器

​	ncvv -Xptxas -v, -abi=no // 输出寄存器的数量， 共享内存字节数， 常量内存字节数

2.共享内存

3.本地内存

​	____share____ 片上内存，在核函数的范围内声明

​	SM中的一级缓存和共享内存都使用64KB的片上内存划分，它通过静态划分，但在运行时可以通过指令进行动态配置

~~~c
cudaError_t cudaFuncSetCacheConfig(const void* func, enum cudaFuncCache cacheConfig);
/* 	func参数
	cudaFuncCachePreferNone: 	没有参考值
	cudaFuncCachePreferShared: 	建议48KB的共享内存和16KB的一级缓存
	cudaFuncCachePreferL1:		建议48KB的一级缓存和16KB的共享内存
	cudaFuncCachePreferEqual:	建议相同尺寸的一级缓存核共享内存， 都是32KB
~~~



4.常量内存

​	——constant——

​	驻留在设备内存中，必须在全局空间内和所有核函数之外进行声明

​	核函数只能从常量内存中读取数据，常量内存必须在主极端使用下面函数初始化：

~~~c
cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count);
/* 	线程束中所有线程从相同的内存地址中读取数据时，常量内存表现最好 
	如果线程束中每个线程都从不同的地址空间读取数据，并且只读一次，常量内存不是最佳选择，因为每从一个常量内存中读取一次数据，都会广播给线程束里的所有线程 */
~~~



5.纹理内存

​	纹理内存驻留在设备内存中，并在每个SM的只读缓存中缓存；纹理内存是一种通过指定的只读缓存访问的全局内存。

​	二维空间线程束使用纹理内存访问二维数据的线程可以达最好的性能

​	对应用程序，纹理内存比全局内存慢

6.全局内存

​	声明可以在任何SM上被访问到，贯穿应用程序的整个生命周期

​	静态声明一个变量：——device——

​	动态：cudaMalloc 使用cudaFree释放

7.GPU缓存

​	一级缓存

​	二级缓存

​	只读常量缓存

​	只读纹理缓存

​	每个SM都有一个一级缓存，所有的SM共享一个二级缓存；每个SM只有一个只读常量/纹理缓存，在设备内存中提高来自于各自内存空间内的读取性能。

![image-20240418234232102](/home/humble/.config/Typora/typora-user-images/image-20240418234232102.png)



~~~c
cudaMemcpyToSymbol(devData, &value, sizeof(float));
cudaMemcpyFromSymbol(&value, devData, sizeof(float));
/* 	获取全局变量地址	*/
cudaError_t cudaGetSymbolAddress(void** devPtr, const void* symbol);
~~~



~~~c
~~~

