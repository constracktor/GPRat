#ifndef TARGET_H
#define TARGET_H

#include <string>

#if GPRAT_WITH_CUDA
#include <cuda_runtime.h>
#include <hpx/async_cuda/cublas_executor.hpp>
#endif

namespace gprat
{

/**
 * @brief This class represents the target on which to perform the Gaussian
 *        Process computations: either CPU or GPU.
 *
 * The respective subclasses implement specific targets: CPU, CUDA_GPU.
 * They may also set additional attributes or function that are required when
 * using this target.
 */
struct Target
{
    /**
     * @brief Returns true if target is CPU.
     *
     * Implemented by subclasses.
     */
    virtual bool is_cpu() = 0;

    /**
     * @brief Returns true if target is GPU.
     *
     * Implemented by subclasses.
     */
    virtual bool is_gpu() = 0;

    /**
     * @brief Returns string representation of the target.
     *
     * Implemented by subclasses.
     */
    virtual std::string repr() const = 0;

    virtual ~Target() { }

  protected:
    Target() = default;
};

struct CPU : public Target
{
  public:
    /**
     * @brief Returns CPU target.
     */
    CPU();

    /**
     * @brief Returns true because target is CPU.
     */
    bool is_cpu() override;

    /**
     * @brief Returns false because CPU target is not GPU.
     */
    bool is_gpu() override;

    /**
     * @brief Returns string representation of the CPU target.
     */
    std::string repr() const override;
};

/**
 * @brief Creates and returns handle for CPU target.
 *
 * @return CPU target
 */
CPU get_cpu();

#if GPRAT_WITH_CUDA
struct CUDA_GPU : public Target
{
    /**
     * @brief Identifier of GPU device.
     *
     * Can be set to a value between 0 and gpu_count().
     */
    int id;

    /**
     * @brief Number of CUDA streams used asynchronous computation and data
     *        transfer.
     */
    int n_streams;

    /**
     * @brief Index of next CUDA stream assigned on next_stream() or
     *        next_cublas_handle().
     */
    int i_stream;

    /** @brief Default amount of CUDA shared memory used by CUDA kernels. */
    int shared_memory_size;

    /**
     * @brief Returns GPU target that uses CUDA.
     */
    CUDA_GPU(int id, int n_streams);

    /**
     * @brief Returns false because target is not CPU.
     */
    bool is_cpu() override;

    /**
     * @brief Returns true because target is GPU.
     */
    bool is_gpu() override;

    /**
     * @brief Returns string representation of the GPU target.
     */
    std::string repr() const override;

    /**
     * @brief Creates n_streams CUDA streams and cublas handles.
     *
     * WARNING: Call destroy() to free both resources after using them.
     */
    void create();

    /**
     * @brief Destroys the CUDA streams and cublas handles previously created
     *        with create().
     */
    void destroy();

    /**
     * @brief Returns the next CUDA streams.
     *
     * It regards the collection of CUDA streams as a cyclic list and returns
     * the next CUDA stream in the cycle. The returned stream was already
     * created when calling create() and will be destroyed by using destroy().
     *
     * @return CUDA stream
     */
    cudaStream_t next_stream();

    /**
     * @brief Synchronizes the collection of CUDA streams.
     *
     * The streams must have be retrieved by next_stream(). Thus, it can use the
     * cyclic ordering to sync each stream in subset_of_streams only once.
     *
     * @param subset_of_streams Vector of CUDA streams, previously retrieved
     *                          with next_stream().
     */
    void sync_streams(std::vector<cudaStream_t> &subset_of_streams);

    /**
     * @brief Returns the next cuBLAS handle.
     *
     * It regards the collection of cuBLAS handles as a cyclic list and returns
     * the next handle in the cycle. The returned handle was already
     * created when calling create() and will be destroyed by using destroy().
     *
     * @return cuBLAS handle
     */
    std::pair<cublasHandle_t, cudaStream_t> next_cublas_handle();

  private:
    std::vector<cudaStream_t> streams;
    std::vector<cublasHandle_t> cublas_handles;
};

/**
 * @brief Creates and returns handle for GPU target.
 *
 * @param id ID of GPU.
 * @param n_streams Number of streams to be created on GPU.
 *
 * @return GPU target
 */
CUDA_GPU get_gpu(int id, int n_streams);

/**
 * @brief Returns handle for GPU target with ID 0.
 *
 * Uses only one stream, so single-threaded GPU execution.
 */
CUDA_GPU get_gpu();
#endif

/**
 * @brief Lists available GPUs with their properties.
 */
void print_available_gpus();

/**
 * @brief Returns number of available GPUs.
 */
int gpu_count();

}  // namespace gprat

#endif  // end of TARGET_H
