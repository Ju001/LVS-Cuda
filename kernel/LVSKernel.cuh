#ifndef LVSKERNEL_H
#define LVSKERNEL_H
#include <map>
#include <array>
#include <string>
#include <vector>
#include <cuda_runtime_api.h>
/**
 * Implements Lowest-Variance Streamlines for Filtering of 3D Ultrasound
 * Runs on Cuda.
 */
namespace LVS
{
	class LVSKernel
	{
	public:
		/*!
		 * Calls the filter method. Is the sole entrypoint for this library.
		 * @param[in] data The volume data provided as flat, normalised float array (0.0f...1.0f).
		 * @param[in] dimensions The dimensions of the provided volume.
		 * @param[in] params Parameters given to kernel as map.
		 */
		std::vector<float> filter(std::vector<float> &data, std::array<int, 3> dimensions, std::map<std::string, int> params);

	private:
		template <class T>
		cudaArray *create3DArray(std::array<int, 3> dimensions);
		template <class T>
		void copyTo3DArray(std::array<int, 3> dimensions, T *data, cudaArray *dstArray);
		cudaTextureObject_t createTextureObject(cudaArray *data);
	};
}
#endif // LVSKernel