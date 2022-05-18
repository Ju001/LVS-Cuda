#define CUDA_VERSION 11060
#define GLM_FORCE_CUDA
#include "LVSKernel.cuh"
#include <iostream>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>
#include <glm/gtx/common.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <math.h>
#include <vector_functions.h>
#include <helper_math.h>
#include <string>
#include <map>
using namespace glm;

namespace LVS
{
/*!
 * Wrapperfunction for CUDA-calls. Prints error to stdout.
 */
#define gpuErrchk(ans)                        \
	{                                         \
		gpuAssert((ans), __FILE__, __LINE__); \
	}
	void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
	{
		if (code != cudaSuccess)
		{
			fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
			if (abort)
				exit(code);
		}
	}

	/*!
	 * Function to get voxel from a 3D volume, saved as flat array.
	 * Returns index.
	 */
	__device__ int getIndex(int x, int y, int z, int width, int height, int depth)
	{
		return x + y * width + z * width * height;
	}

	/*!
	 * Finds the direction with the least variance for every voxel and writes it in a 3D vectorfield.
	 * @param[in] volume The volume data provided as cudaTextureObject_t.
	 * @param[in] width,height,depth The dimensions of the provided volume.
	 * @param[in] samples The amount of samples used to calculate the variance in every direction.
	 * @param[in] sampleAngle The angle between each sampled direction (degrees).
	 * @param[out] output The 3D vectorfield provided as flat, normalised float4 array.
	 */
	__global__ void createVectorField(cudaTextureObject_t volume, int width, int height, int depth, int samples, int sampleAngle, float4 *output)
	{
		// Index of voxel the current thread is responsible for (https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2017/01/cuda_indexing.png)
		int indexX = blockIdx.x * blockDim.x + threadIdx.x;
		int indexY = blockIdx.y * blockDim.y + threadIdx.y;
		int indexZ = blockIdx.z * blockDim.z + threadIdx.z;
		// Stop if indizes should exceed volume
		if (indexX >= width || indexY >= height || indexZ >= depth)
		{
			return;
		}
		// Stride if not enough threads for every voxel available
		int strideX = blockDim.x * gridDim.x;
		int strideY = blockDim.y * gridDim.y;
		int strideZ = blockDim.z * gridDim.z;

		// Amount of angles sampled in two planes, but only half of the hemisphere
		int amountDirections = ((360 / sampleAngle) * (360 / sampleAngle)) / 2;
		int circleSteps = 360 / sampleAngle;
		float minVariance;
		// Holds the direction with minimum variance. Gets written into vectorfield.
		float3 minDirection;
		for (int x = indexX; x < width; x += strideX)
		{
			for (int y = indexY; y < height; y += strideY)
			{
				for (int z = indexZ; z < depth; z += strideZ)
				{
					// Repeat for every sample direction
					for (int i = 0; i < amountDirections; i++)
					{

						// Calculate new direction
						fvec3 newDirection = fvec3(.0f, .0f, .0f);
						// Vector is rotated in two planes.
						int angleXY = (i / circleSteps) * sampleAngle;
						int angleXZ = (i % circleSteps) * sampleAngle;
						newDirection = rotate(fvec3(1, 0, 0), radians((float)angleXY), fvec3(0, 0, 1));
						newDirection = rotate(newDirection, radians((float)angleXZ), fvec3(0, 1, 0));

						// Calculating variance (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Computing_shifted_data)
						fvec3 point;
						point.x = x, point.y = y, point.z = z;
						float n = 0;
						float sum = 0;
						float sum_sqr = 0;
						float dataK = 0;

						// Negative direction
						for (int k = 0; k < samples; k++)
						{
							point = point - newDirection;
							// Have to add 0.5f because of interpolation
							float data = tex3D<float>(volume, point.x + 0.5f, point.y + 0.5f, point.z + 0.5f);
							if (k == 0)
							{
								dataK = data;
							}
							n = n + 1;
							sum += data - dataK;
							sum_sqr += (data - dataK) * (data - dataK);
						}

						// Positive direction
						point.x = x, point.y = y, point.z = z;
						for (int k = 0; k < samples; k++)
						{
							point = point + newDirection;
							// Have to add 0.5f because of interpolation
							float data = tex3D<float>(volume, point.x + 0.5f, point.y + 0.5f, point.z + 0.5f);
							if (k == 0)
							{
								dataK = data;
							}
							n = n + 1;
							sum += data - dataK;
							sum_sqr += (data - dataK) * (data - dataK);
						}

						float variance = (sum_sqr - ((sum * sum) / n)) / n;
						// Initialise minVariance if first pass.
						if (i > 0)
						{
							minVariance = glm::min(variance, minVariance);
							if (minVariance == variance)
							{
								minDirection = make_float3(newDirection.x, newDirection.y, newDirection.z);
							}
						}
						else
						{
							minVariance = variance;
							minDirection = make_float3(newDirection.x, newDirection.y, newDirection.z);
						}
						// Stop if variance is 0
						if (minVariance <= 0.0f + FLT_EPSILON)
						{
							break;
						}
					}
					// Write direction with lowest variance to output
					float4 erg = make_float4(minDirection.x, minDirection.y, minDirection.z, 0.0f);
					output[getIndex(x, y, z, width, height, depth)] = erg;
				}
			}
		}
	}
	/*!
	 * http://graphics.cs.ucdavis.edu/~joy/ecs277/other-notes/Numerical-Methods-for-Particle-Tracing-in-Vector-Fields.pdf
	 * Runge-Kutta integration in the positive vectorfield direction.
	 * @param[in] vectorField The vectorfield as cudaTextureObject_t.
	 * @param[in] point Seedpoint as float4 (CUDA can't create textures with float3).
	 */
	__device__
		float4
		rungeKutta(cudaTextureObject_t vectorField, float4 point)
	{
		float4 k1, k2, k3, k4, vTemp;

		k1 = tex3D<float4>(vectorField, point.x + .5f, point.y + .5f, point.z + .5f);
		vTemp = point + (.5f * k1);
		k2 = tex3D<float4>(vectorField, vTemp.x + .5f, vTemp.y + .5f, vTemp.z + .5f);
		vTemp = point + (.5f * k2);
		k3 = tex3D<float4>(vectorField, vTemp.x + .5f, vTemp.y + .5f, vTemp.z + .5f);
		vTemp = point + (.5f * k3);
		k4 = tex3D<float4>(vectorField, vTemp.x + .5f, vTemp.y + .5f, vTemp.z + .5f);

		return point + 0.166666667 * (k1 + 2.0f * k2 + 2.0f * k3 + k4);
	}
	/*!
	 * http://graphics.cs.ucdavis.edu/~joy/ecs277/other-notes/Numerical-Methods-for-Particle-Tracing-in-Vector-Fields.pdf
	 * Runge-Kutta integration in the negative vectorfield direction.
	 * @param[in] vectorField The vectorfield as cudaTextureObject_t.
	 * @param[in] point Seedpoint as float4 (CUDA can't create textures with float3).
	 */
	__device__
		float4
		invertedRungeKutta(cudaTextureObject_t vectorField, float4 point)
	{
		float4 k1, k2, k3, k4, vTemp;

		k1 = tex3D<float4>(vectorField, point.x + .5f, point.y + .5f, point.z + .5f) * -1.0f;
		vTemp = point + (.5f * k1);
		k2 = tex3D<float4>(vectorField, vTemp.x + .5f, vTemp.y + .5f, vTemp.z + .5f) * -1.0f;
		vTemp = point + (.5f * k2);
		k3 = tex3D<float4>(vectorField, vTemp.x + .5f, vTemp.y + .5f, vTemp.z + .5f) * -1.0f;
		vTemp = point + (.5f * k3);
		k4 = tex3D<float4>(vectorField, vTemp.x + .5f, vTemp.y + .5f, vTemp.z + .5f) * -1.0f;

		return point + 0.166666667f * (k1 + 2.0f * k2 + 2.0f * k3 + k4);
	}

	/*!
	 * Integrates vectorfield in the positive and negative directions and calculates the arithmetic mean
	 * on the corresponding volume positions for a given seedpoint.
	 * @param[in] volume The volume data provided as cudaTextureObject_t.
	 * @param[in] width,height,depth The dimensions of the provided volume.
	 * @param[in] steps The amount of integration steps to be computed.
	 * @param[out] output The filtered volume data provided as flat, normalised float array (0.0f...1.0f).
	 */
	__global__ void integrateVectorField(cudaTextureObject_t volume, cudaTextureObject_t vectorField, int width, int height, int depth, int steps, float *output)
	{
		// Index of voxel the current thread is responsible for (https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2017/01/cuda_indexing.png)
		int indexX = blockIdx.x * blockDim.x + threadIdx.x;
		int indexY = blockIdx.y * blockDim.y + threadIdx.y;
		int indexZ = blockIdx.z * blockDim.z + threadIdx.z;
		// Stop if indizes should exceed volume
		if (indexX >= width || indexY >= height || indexZ >= depth)
		{
			return;
		}
		// Stride if not enough threads for every voxel available
		int strideX = blockDim.x * gridDim.x;
		int strideY = blockDim.y * gridDim.y;
		int strideZ = blockDim.z * gridDim.z;

		for (int x = indexX; x < width; x += strideX)
		{
			for (int y = indexY; y < height; y += strideY)
			{
				for (int z = indexZ; z < depth; z += strideZ)
				{
					float4 currentPoint = make_float4(x, y, z, 0.0f);
					// Initialise sum with the seedpoint
					float sum = tex3D<float>(volume, currentPoint.x + 0.5f, currentPoint.y + 0.5f, currentPoint.z + 0.5f);

					// Integrating in the negative direction
					for (int intStep = 0; intStep < steps; intStep++)
					{
						currentPoint = invertedRungeKutta(vectorField, currentPoint);
						sum += tex3D<float>(volume, currentPoint.x + 0.5f, currentPoint.y + 0.5f, currentPoint.z + 0.5f);
					}
					// Integrating in the positive direction
					currentPoint = make_float4(x, y, z, 0.0f);
					for (int intStep = 0; intStep < steps; intStep++)
					{
						currentPoint = rungeKutta(vectorField, currentPoint);
						sum += tex3D<float>(volume, currentPoint.x + 0.5f, currentPoint.y + 0.5f, currentPoint.z + 0.5f);
					}
					// Writing back the arithmetic mean
					output[getIndex(x, y, z, width, height, depth)] = (1.0f / (2.0f * (float)steps + 1.0f)) * sum;
				}
			}
		}
	}
	template <class T>
	cudaArray *LVSKernel::create3DArray(std::array<int, 3> dimensions)
	{
		cudaArray *cuda3DArray;
		cudaChannelFormatDesc formatDesc = cudaCreateChannelDesc<T>();
		cudaExtent cudaSize = make_cudaExtent(dimensions[0], dimensions[1], dimensions[2]);
		gpuErrchk(cudaMalloc3DArray(&cuda3DArray, &formatDesc, cudaSize, 0));
		return cuda3DArray;
	}

	template <class T>
	void LVSKernel::copyTo3DArray(std::array<int, 3> dimensions, T *data, cudaArray *dstArray)
	{
		cudaExtent cudaSize = make_cudaExtent(dimensions[0], dimensions[1], dimensions[2]);
		cudaMemcpy3DParms copyParams = {0};
		copyParams.srcPtr = make_cudaPitchedPtr(data, dimensions[0] * sizeof(T), dimensions[0], dimensions[1]);
		copyParams.dstArray = dstArray;
		copyParams.extent = cudaSize;
		copyParams.kind = cudaMemcpyHostToDevice;
		gpuErrchk(cudaMemcpy3D(&copyParams));
	}

	cudaTextureObject_t LVSKernel::createTextureObject(cudaArray *data)
	{
		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = data;
		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.readMode = cudaReadModeElementType;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.addressMode[2] = cudaAddressModeWrap;
		cudaTextureObject_t tex = 0;
		gpuErrchk(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
		return tex;
	}

	std::vector<float> LVSKernel::filter(std::vector<float> &data, std::array<int, 3> dimensions, std::map<std::string, int> params)
	{
		int samples = params["samples"];
		int integrationSteps = params["integrationSteps"];
		int sampleAngle = params["sampleAngles"];
		int width = dimensions[0];
		int height = dimensions[1];
		int depth = dimensions[2];

		// Calculate vectorfield
		printf("Calculate vectorfield...\n");
		// Amount of angles sampled in two planes, but only half of the hemisphere
		int amountDirections = (pow((360 / sampleAngle), 2)) / 2;
		int size = width * height * depth;
		// Outputpointer for vectorfield. Use float4 because CUDA doesn't know float3 textures.
		float4 *vectorfield;
		gpuErrchk(cudaMallocManaged(&vectorfield, sizeof(float4) * size));

		// Create 3D array to hold the volume data
		cudaArray *volumeBuffer = create3DArray<float>(dimensions);

		// Copy volumedata to 3D array
		copyTo3DArray<float>(dimensions, &data.front(), volumeBuffer);

		// Create texture object. Using texture instead of raw pointer gives us interpolation for free
		cudaTextureObject_t volumeBufferTex = createTextureObject(volumeBuffer);

		// Execute createVectorField
		// Blocksize optimized for sample data(64 threads per block)
		dim3 blockSizeVF(8, 8, 1);
		// Calculate number of blocks needed with given blocksize (blockSizeVF. - 1 is for rounding up)
		dim3 gridSizeVF((width + blockSizeVF.x - 1) / blockSizeVF.x, (height + blockSizeVF.y - 1) / blockSizeVF.y,
						((depth + blockSizeVF.z - 1) / blockSizeVF.z));
		createVectorField<<<gridSizeVF, blockSizeVF>>>(volumeBufferTex, width, height, depth, samples, sampleAngle, vectorfield);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		// Integrate Vector field
		printf("Filtering data...\n");

		// Create 3D array to hold vectorfieldbuffer
		cudaArray *vectorFieldBuffer = create3DArray<float4>(dimensions);

		// copy data to 3D array
		copyTo3DArray<float4>(dimensions, vectorfield, vectorFieldBuffer);

		// create texture object;
		cudaTextureObject_t vectorFieldtex = createTextureObject(vectorFieldBuffer);

		// Prepare outputbuffer
		float *filtered;
		gpuErrchk(cudaMallocManaged(&filtered, sizeof(float) * size));

		// Execute filtering
		// Blocksize optimized for sample data(64 threads per block)
		dim3 blockSizeFilter(8, 8, 1);
		// Calculate number of blocks needed with given blocksize (blockSizeFilter. - 1 is for rounding up)
		dim3 gridSizeFilter((width + blockSizeFilter.x - 1) / blockSizeFilter.x, (height + blockSizeFilter.y - 1) / blockSizeFilter.y,
							((depth + blockSizeVF.z - 1) / blockSizeVF.z));
		integrateVectorField<<<gridSizeFilter, blockSizeFilter>>>(volumeBufferTex, vectorFieldtex, width, height, depth, integrationSteps, filtered);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		std::vector<float> output = std::vector<float>(sizeof(float) * size, 0);
		// Copy result to output
		memcpy(&output.front(), filtered, sizeof(float) * size);

		gpuErrchk(cudaFree(vectorfield));
		gpuErrchk(cudaFreeArray(volumeBuffer));
		gpuErrchk(cudaFreeArray(vectorFieldBuffer));
		gpuErrchk(cudaDestroyTextureObject(volumeBufferTex));
		gpuErrchk(cudaDestroyTextureObject(vectorFieldtex));
		gpuErrchk(cudaFree(filtered));
		return output;
	}
}