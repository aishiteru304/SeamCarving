#include <stdio.h>
#include <stdint.h>
#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}
void printDeviceInfo()
{
	cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("CMEM: %lu bytes\n", devProv.totalConstMem);
    printf("L2 cache: %i bytes\n", devProv.l2CacheSize);
    printf("SMEM / one SM: %lu bytes\n", devProv.sharedMemPerMultiprocessor);

    printf("****************************\n");

}
struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};
void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	
	if (strcmp(type, "P3") != 0) // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);
	
	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	for (int i = 0; i < width * height; i++)
		fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

	fclose(f);
}
void writePnm(uchar3 * pixels, int width, int height, char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	fprintf(f, "P3\n%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height; i++)
		fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);
	
	fclose(f);
}
#define FILTER_WIDTH 3
__constant__ float dc_filter[FILTER_WIDTH * FILTER_WIDTH];
__global__ void Convolution_Kernel(uchar3 * inPixels, int width, int height, 
        int filterWidth, 
        uchar3 * outPixels)
{
	//This shared memory contain all memory need to use for one block from global memory
	extern __shared__ uchar3 s_inPixels[];
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (x <= (width - 1) && y <= (height - 1))
	{
		//New size after padding
		int Cols = blockDim.x + filterWidth - 1;
		int Rows = blockDim.y + filterWidth - 1;
		// Copy to share memory all data needed for the block after padding
		for (int i = 0; i < Cols - threadIdx.x; i += blockDim.x)
		{
			for (int j = 0; j < Rows - threadIdx.y; j += blockDim.y)
			{
				int relative_x = i + x - filterWidth/2 > 0 ? i + x - filterWidth/2 : 0;
				int relative_y = j + y - filterWidth/2 > 0 ? j + y - filterWidth/2 : 0;
				if (relative_x > width - 1)
					relative_x = width - 1;
				if (relative_y > height -1)
					relative_y = height - 1;
				int idx = Cols * (threadIdx.y+j) + (threadIdx.x + i);
				s_inPixels[idx] = inPixels[relative_x + relative_y * width];	
			}
		}
		// Synchronize
		__syncthreads();

    float3 res = make_float3(0, 0, 0);
		// Loop through kernel
		for (int i = 0; i < filterWidth; i++) 
		{
			for (int j = 0; j < filterWidth; j++)
			{
				float reg = dc_filter[i*filterWidth+j];
				uchar3  value = s_inPixels[threadIdx.x + j + (threadIdx.y + i) * Cols];
				res.x += reg * value.x;
				res.y += reg * value.y;
				res.z += reg * value.z;
			}
		}

		outPixels[x + y * width] =  make_uchar3(fabs(res.x), fabs(res.y), fabs(res.z));

	}
}
__global__ void addMatKernel(uchar3 *in1, uchar3 *in2, int nRows, int nCols, 
        double *out)
{
    // TODO
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < nRows && c < nCols)
    { 
        int i = r * nCols + c;
        out[i] = (in1[i].x+in1[i].y+in1[i].z) + (in2[i].x+in2[i].y+in2[i].z);
    }
}
void Convolution(uchar3 * inPixels, int width, int height, float * filter, int filterWidth, 
        uchar3 * outPixels, dim3 blockSize=dim3(1, 1))
{
    size_t pixelsSize = width * height * sizeof(uchar3); 
    // Initilal in devices
    uchar3 * d_inPixels, * d_outPixels;
    CHECK(cudaMalloc(&d_inPixels, pixelsSize));
    CHECK(cudaMalloc(&d_outPixels, pixelsSize));

    size_t filterSize = filterWidth * filterWidth * sizeof(float);
    
    cudaMemcpyToSymbol(dc_filter, filter, filterSize, 0, cudaMemcpyHostToDevice);
    CHECK(cudaMemcpy(d_inPixels, inPixels, pixelsSize, cudaMemcpyHostToDevice));
    
    dim3 gridSize((width-1)/blockSize.x + 1, (height-1)/blockSize.y + 1);
    int Cols = blockSize.x + filterWidth - 1;
    int Rows = blockSize.y + filterWidth - 1;
    // Call the kernel function
    Convolution_Kernel<<<gridSize, blockSize, Cols * Rows * sizeof(uchar3)>>>(d_inPixels, width, height, filterWidth, d_outPixels);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(outPixels, d_outPixels, pixelsSize, cudaMemcpyDeviceToHost));
    
    CHECK(cudaFree(d_inPixels));
    CHECK(cudaFree(d_outPixels));
}
static double* getEnergy(uchar3* pixels, int width, int height, bool debug = 0)
{
    float * filter = (float *)malloc(3 * 3 * sizeof(float));
    //Convolution Sobel |Di/Dx|
    uchar3* Dx = (uchar3 *)malloc(width * height * sizeof(uchar3));
    filter[0]=1;filter[1]=0;filter[2]=-1;filter[3]=2;filter[4]=0;filter[5]=-2;filter[6]=1;filter[7]=0;filter[8]=-1;
    Convolution(pixels, width, height, filter, 3, Dx, dim3(32,32));
  
    //Convolution Sobel |Di/Dy|
    uchar3* Dy = (uchar3 *)malloc(width * height * sizeof(uchar3));
    filter[0]=1;filter[1]=2;filter[2]=1;filter[3]=0;filter[4]=0;filter[5]=0;filter[6]=-1;filter[7]=-2;filter[8]=-1;
    Convolution(pixels, width, height, filter, 3, Dy, dim3(32,32));
 
    if (debug == 1)
    {
      //Sum 3 channel and Dx and Dy to get energy
      uchar3* DebugEnergy = (uchar3*) malloc(width * height * sizeof(uchar3));
      
      for (int i = 0; i < width * height; i++)
      {
        int average = (((Dx[i].x) + (Dy[i].x))/2 + ((Dx[i].y) + (Dy[i].y))/2 + ((Dx[i].z) + (Dy[i].z))/2)/3;
        DebugEnergy[i].x = average;
        DebugEnergy[i].y = average;
        DebugEnergy[i].z = average;
      }
     
      writePnm(Dx, width, height, "Dx.pnm");
      writePnm(Dy, width, height, "Dy.pnm");
      writePnm(DebugEnergy, width, height, "energies.pnm");
      free(DebugEnergy);

    }
 
    double* energies = (double*) malloc(width * height * sizeof(double));
 
    // Add vector /Dx/+/Dy/
    uchar3* device_Dx; 
    uchar3* device_Dy;
    double* device_energies;
    //Device allocate memory
    CHECK(cudaMalloc(&device_Dx, width*height*sizeof(uchar3)));
    CHECK(cudaMalloc(&device_Dy, width*height*sizeof(uchar3)));
    CHECK(cudaMalloc(&device_energies, width*height*sizeof(double)));

    //Copy Dx and Dy image to device
    CHECK(cudaMemcpy(device_Dx, Dx, width*height*sizeof(uchar3), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_Dy, Dy, width*height*sizeof(uchar3), cudaMemcpyHostToDevice));

    //Setup blockSize and Gridsize
    dim3 blockSize=dim3(32, 32);
    dim3 gridSize = dim3((width-1)/blockSize.x+1,(height-1)/blockSize.y+1);
  
    //Calculate /Dx/+/Dy/
    addMatKernel<<<gridSize, blockSize>>>(device_Dx, device_Dy, width, height, device_energies);
 
    cudaDeviceSynchronize();
    CHECK(cudaMemcpy(energies, device_energies, width*height*sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaGetLastError())
    CHECK(cudaPeekAtLastError());

    free(Dx);
    free(Dy);
    free(filter);
    return energies;
  
}
int FindIndexOfMin(double* a, int left, int right)
{
    int index_of_min = left;
    double min_val = a[left];
    for(int i = left; i <= right; i++)
    {
        if (min_val > a[i])
           {
               min_val = a[i];
               index_of_min = i;
           }
    }
    return index_of_min;
}
__global__ void Calculate_Cumulative(double* energies,  int width, int height, double* cumulative, int* direction)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // This is index of column 0-->1024
  if (idx < width)
  {
      cumulative[idx] = energies[idx];
  }
  __syncthreads();
  for(int j = 1; j < height; j++)
  {
      if(idx == 0)
      {
          double mid = cumulative[(j-1)*width+idx];
          double right = cumulative[(j-1)*width+idx+1];
          if (mid < right)
          {
              cumulative[j*width+idx] = mid + energies[j*width+idx];
              direction[j*width+idx] = 1;
          }
          else
          {
              cumulative[j*width+idx] = right + energies[j*width+idx];
              direction[j*width+idx] = 2;
          }
      }
      else
      {
          if(idx == width -1)
          {
              double left = cumulative[(j-1)*width+idx-1];
              double mid = cumulative[(j-1)*width+idx];
              if (left < mid)
              {
                  cumulative[j*width+idx] = left + energies[j*width+idx];
                  direction[j*width+idx] = 0;
              }
              else
              {
                  cumulative[j*width+idx] = mid + energies[j*width+idx];
                  direction[j*width+idx] = 1;
              }
          }
          else
        {
             double left = cumulative[(j-1)*width+idx-1];
             double mid = cumulative[(j-1)*width+idx];
             double right = cumulative[(j-1)*width+idx+1];
             if (left <= mid && left <= right)
              {
                  cumulative[j*width+idx] = left + energies[j*width+idx];
                  direction[j*width+idx] = 0;
              }
             else
              { 
                  if(mid <= left && mid <= right)
                    {
                      cumulative[j*width+idx] = mid + energies[j*width+idx];
                      direction[j*width+idx] = 1;
                    }
                  else 
                   {
                       cumulative[j*width+idx] = right + energies[j*width+idx];
                       direction[j*width+idx] = 2;
                   }
              }
        }
      }
   
   __syncthreads();
  }
  
} 

static int*  get_route_of_minimum_cumulative_enery(const double* energies, int width, int height) {
    // cumulative enery
    double* cumulative =(double* ) malloc(width * height * sizeof(double));
    if (cumulative == NULL) return NULL;
    // 0: above-left   1: above    2: above-right
    int* direction =(int* ) calloc(width * height, sizeof(int));
    if (direction == NULL) return NULL;

    double * device_energies;
    double * device_cumulative;
    int* device_direction;

    CHECK(cudaMalloc(&device_energies, width*height*sizeof(double)));
    CHECK(cudaMemcpy(device_energies, energies, width*height*sizeof(double), cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(&device_cumulative, width*height*sizeof(double)));
    CHECK(cudaMalloc(&device_direction, width*height*sizeof(int)));

  
    dim3 blockSize=dim3(min(1024, width));
    dim3 gridSize = dim3(1);
    
 
  
    Calculate_Cumulative<<<gridSize, blockSize>>>(device_energies, width, height, device_cumulative, device_direction);
    cudaDeviceSynchronize();
    CHECK(cudaMemcpy(cumulative, device_cumulative, width*height*sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(direction, device_direction, width*height*sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaGetLastError());
    CHECK(cudaPeekAtLastError());

    
    int col = FindIndexOfMin(cumulative, (height - 1) * width + 0, (height - 1) * width + width-1) - ((height - 1) * width);
    free(cumulative);

    int* route = (int* )malloc(height * sizeof(int));
    if (route == NULL) return NULL;
    for (int r = height-1; r >= 0; r--) 
    {
        route[r] = col;
        int move = direction[r * width + col];
        if(direction[r * width + col] == 0) col--;
        else if(direction[r * width + col] == 2) col++;
    }
    free(direction);
    return route;
}
static void resize_image(uchar3* pixels, int width, int height, int num_of_column_deleted)
{
  uchar3* image_before = pixels;
  double* energies = (double*) malloc(height * width * sizeof(double));
  energies = getEnergy(pixels, width, height, 1);

 for (int i = 1; i <= num_of_column_deleted; i++)
 {
    int new_width = width - 1;
    uchar3* resize_remove_one_column_image = (uchar3 *) malloc(height*new_width*sizeof(uchar3));
    double* new_energies = (double* ) malloc(height * new_width * sizeof(double));
    int* min_route = get_route_of_minimum_cumulative_enery(energies, width, height);
    // Remove pixels
    for (int i = 0; i < height; i++) 
    {
      int index_of_resize_image = 0; // New j index after the offset
      for (int j = 0; j < width; j++) 
      {
          if (j != min_route[i])
          {
            resize_remove_one_column_image[i * new_width + index_of_resize_image] = image_before[i * width + j];
            index_of_resize_image++;
          }
      }
    }
    width--;
    new_energies = getEnergy(resize_remove_one_column_image, new_width, height);
    free(energies); free(min_route);
    image_before = resize_remove_one_column_image;
    energies = new_energies;
 }
 writePnm(image_before, width, height, "result.pnm");

}

int main(int argc, char ** argv)
{
  
  printDeviceInfo();
  printf("\n ------------------------------------------------------ \n");
  uchar3* inPixels;
  int width, height;
  readPnm(argv[1], width, height, inPixels);
  
  GpuTimer timer;
  timer.Start();
  resize_image(inPixels, width, height, atoi(argv[2]));
  timer.Stop();
  float total_time = timer.Elapsed();
  printf("Resize Image take: %f", total_time);
}
