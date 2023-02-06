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
void Convolution(uchar3 * inPixels, int width, int height, float * filter, int filterWidth, 
        uchar3 * outPixels,
        bool useDevice=false, dim3 blockSize=dim3(1, 1), int kernelType=1)
{

		for (int outPixelsR = 0; outPixelsR < height; outPixelsR++)
		{
			for (int outPixelsC = 0; outPixelsC < width; outPixelsC++)
			{
				float3 outPixel = make_float3(0, 0, 0);
				for (int filterR = 0; filterR < filterWidth; filterR++)
				{
					for (int filterC = 0; filterC < filterWidth; filterC++)
					{
						float filterVal = filter[filterR*filterWidth + filterC];
						int inPixelsR = outPixelsR - filterWidth/2 + filterR;
						int inPixelsC = outPixelsC - filterWidth/2 + filterC;
						inPixelsR = min(max(0, inPixelsR), height - 1);
						inPixelsC = min(max(0, inPixelsC), width - 1);
						uchar3 inPixel = inPixels[inPixelsR*width + inPixelsC];
						outPixel.x += filterVal * inPixel.x;
						outPixel.y += filterVal * inPixel.y;
						outPixel.z += filterVal * inPixel.z;
					}
				}
				outPixels[outPixelsR*width + outPixelsC] = make_uchar3(fabs(outPixel.x), fabs(outPixel.y), fabs(outPixel.z)); 
			}
		}
	}
static double* getEnergy(uchar3* pixels, int width, int height, bool debug = 0)
{
    float * filter = (float *)malloc(3 * 3 * sizeof(float));
  //Convolution Sobel |Di/Dx|
  uchar3* Dx = (uchar3 *)malloc(width * height * sizeof(uchar3));
	filter[0]=1;filter[1]=0;filter[2]=-1;filter[3]=2;filter[4]=0;filter[5]=-2;filter[6]=1;filter[7]=0;filter[8]=-1;
	Convolution(pixels, width, height, filter, 3, Dx);
	//Convolution Sobel |Di/Dy|
	uchar3* Dy = (uchar3 *)malloc(width * height * sizeof(uchar3));
	filter[0]=1;filter[1]=2;filter[2]=1;filter[3]=0;filter[4]=0;filter[5]=0;filter[6]=-1;filter[7]=-2;filter[8]=-1;
	Convolution(pixels, width, height, filter, 3, Dy);
  //Sum 3 channel and Dx and Dy to get energy
 	uchar3* DebugEnergy = (uchar3*) malloc(width * height * sizeof(uchar3));
	for (int i = 0; i < width * height; i++)
	{
    int average = (((Dx[i].x) + (Dy[i].x))/2 + ((Dx[i].y) + (Dy[i].y))/2 + ((Dx[i].z) + (Dy[i].z))/2)/3;
		DebugEnergy[i].x = average;
		DebugEnergy[i].y = average;
		DebugEnergy[i].z = average;
	}
 
  if (debug == 1)
  {
      writePnm(Dx, width, height, "Dx.pnm");
      writePnm(Dy, width, height, "Dy.pnm");
      writePnm(DebugEnergy, width, height, "energies.pnm");
  }
	double* energies = (double*) malloc(width * height * sizeof(double));
	for (int i = 0; i < width * height; i++)
	{
		energies[i] = ((Dx[i].x)+(Dx[i].y)+(Dx[i].z))+((Dy[i].x)+(Dy[i].y)+(Dy[i].z));
	}
 free(Dx);
 free(Dy);
 free(DebugEnergy);
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
static int*  get_route_of_minimum_cumulative_enery(const double* energies, int width, int height) {
    // cumulative enery
    double* cumulative =(double* ) malloc(width * height * sizeof(double));
    if (cumulative == NULL) return NULL;
    // 0: above-left   1: above    2: above-right
    int* direction =(int* ) calloc(width * height, sizeof(int));
    if (direction == NULL) return NULL;
    // Cumulative of the first line in the top: Just copy the enery 
    for (int k = 0; k < width; k++)
        cumulative[k] = energies[k];
    
    
    for (int i = 1; i < height; i++) {
        for (int j = 0; j < width; j++) 
        {
            int direction_val;
            if (j == 0) //Leftmost column
            { 
              direction_val = (cumulative[(i-1)*width] < cumulative[(i-1)*width+1])?1:2;
            } 
            else if (j == width - 1) //Rightmost columnn
            { 
              direction_val = (cumulative[(i-1)*width+(j-1)]  < cumulative[(i-1)*width+j])?0:1;
            } 
            else //All of the middle column
            {  // Array of 3 pixel above the current pixel we are target
                double a[3] = {cumulative[(i-1)*width+(j-1)], cumulative[(i-1)*width+j], cumulative[(i-1)*width+(j+1)]};
                direction_val = FindIndexOfMin(a, 0, 2);
            }
            direction[i*width+j] = direction_val;
            cumulative[i*width+j] = cumulative[(i-1)*width+j+direction_val-1]+energies[i * width + j];
        }
    }
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
    // Update pixel energies adjacent to the deleted path
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
