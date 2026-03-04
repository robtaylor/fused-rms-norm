#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <torch/torch.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <string>

// Include the auto-generated header with embedded metallib.
#ifdef EMBEDDED_METALLIB_HEADER
#include EMBEDDED_METALLIB_HEADER
#else
#error "EMBEDDED_METALLIB_HEADER not defined"
#endif

static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor &tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

// Helper to select kernel name by dtype.
static NSString *kernelNameForDtype(const char *prefix,
                                    torch::ScalarType dtype) {
  switch (dtype) {
  case torch::kFloat:
    return [NSString stringWithFormat:@"%s_float", prefix];
  case torch::kHalf:
    return [NSString stringWithFormat:@"%s_half", prefix];
  case torch::kBFloat16:
    return [NSString stringWithFormat:@"%s_bfloat16_t", prefix];
  default:
    TORCH_CHECK(false, "Unsupported dtype: ", dtype);
    return nil;
  }
}

// Helper to load embedded metallib and create pipeline state.
static id<MTLComputePipelineState>
createPipeline(id<MTLDevice> device, NSString *kernName, NSError **error) {
  id<MTLLibrary> lib =
      EMBEDDED_METALLIB_NAMESPACE::createLibrary(device, error);
  TORCH_CHECK(lib, "Failed to create Metal library from embedded data",
              *error ? [NSString stringWithFormat:@": %@",
                                                  (*error).localizedDescription]
                           .UTF8String
                     : "");

  id<MTLFunction> fn = [lib newFunctionWithName:kernName];
  TORCH_CHECK(fn, "Missing Metal kernel function: ", kernName.UTF8String);

  return [device newComputePipelineStateWithFunction:fn error:error];
}

// Dispatch a normalization kernel with 7 buffer bindings + threadgroup memory.
//
// Uses stream->commandEncoder() to properly integrate with PyTorch's MPS
// encoder lifecycle management. The stream tracks the active encoder and
// ends it during synchronize() via endKernelCoalescing().
static void dispatchNormKernel(id<MTLComputePipelineState> pso,
                               at::mps::MPSStream *stream,
                               // Buffers 0-2: tensor buffers with offsets
                               id<MTLBuffer> buf0, NSUInteger off0,
                               id<MTLBuffer> buf1, NSUInteger off1,
                               id<MTLBuffer> buf2, NSUInteger off2,
                               // Scalars
                               float epsilon, int32_t num_tokens,
                               int32_t hidden_size, int64_t input_stride,
                               // Grid
                               uint32_t threadgroups,
                               uint32_t threads_per_tg) {
  // Shared memory: MAX_SIMDGROUPS (16) floats for reduction.
  const uint32_t shared_mem_size = 16 * sizeof(float);

  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
    TORCH_CHECK(enc, "Failed to create compute encoder");

    [enc setComputePipelineState:pso];

    [enc setBuffer:buf0 offset:off0 atIndex:0];
    [enc setBuffer:buf1 offset:off1 atIndex:1];
    [enc setBuffer:buf2 offset:off2 atIndex:2];

    [enc setBytes:&epsilon length:sizeof(float) atIndex:3];
    [enc setBytes:&num_tokens length:sizeof(int32_t) atIndex:4];
    [enc setBytes:&hidden_size length:sizeof(int32_t) atIndex:5];
    [enc setBytes:&input_stride length:sizeof(int64_t) atIndex:6];

    [enc setThreadgroupMemoryLength:shared_mem_size atIndex:0];

    MTLSize grid = MTLSizeMake(threadgroups, 1, 1);
    MTLSize tg = MTLSizeMake(threads_per_tg, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    // Don't call [enc endEncoding] — the stream manages encoder lifecycle
    // via endKernelCoalescing() during synchronize().
  });

  stream->synchronize(at::mps::SyncType::COMMIT_AND_CONTINUE);
}

void rms_norm(torch::Tensor &out, torch::Tensor &input,
              torch::Tensor &weight, double epsilon) {
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(input.device().is_mps(), "input must be on MPS device");

  const int hidden_size = input.size(-1);
  const int64_t input_stride = input.stride(-2);
  const int num_tokens =
      static_cast<int>(input.numel() / hidden_size);

  @autoreleasepool {
    at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(stream, "Failed to get MPS stream");

    id<MTLDevice> device = stream->device();
    NSError *error = nil;

    NSString *kernName = kernelNameForDtype("rms_norm", input.scalar_type());
    id<MTLComputePipelineState> pso = createPipeline(device, kernName, &error);
    TORCH_CHECK(pso, "Pipeline creation failed",
                error ? [NSString stringWithFormat:@": %@",
                                                   error.localizedDescription]
                            .UTF8String
                      : "");

    const uint32_t threads_per_tg =
        std::min<uint32_t>(512, hidden_size);

    dispatchNormKernel(
        pso, stream,
        getMTLBufferStorage(out),
        out.storage_offset() * out.element_size(),
        getMTLBufferStorage(input),
        input.storage_offset() * input.element_size(),
        getMTLBufferStorage(weight),
        weight.storage_offset() * weight.element_size(),
        static_cast<float>(epsilon), static_cast<int32_t>(num_tokens),
        static_cast<int32_t>(hidden_size), input_stride,
        static_cast<uint32_t>(num_tokens), threads_per_tg);
  }
}

void fused_add_rms_norm(torch::Tensor &input, torch::Tensor &residual,
                        torch::Tensor &weight, double epsilon) {
  TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(input.device().is_mps(), "input must be on MPS device");
  TORCH_CHECK(input.scalar_type() == residual.scalar_type(),
              "input and residual must have same dtype");

  const int hidden_size = input.size(-1);
  const int64_t input_stride = input.stride(-2);
  const int num_tokens =
      static_cast<int>(input.numel() / hidden_size);

  @autoreleasepool {
    at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(stream, "Failed to get MPS stream");

    id<MTLDevice> device = stream->device();
    NSError *error = nil;

    NSString *kernName =
        kernelNameForDtype("fused_add_rms_norm", input.scalar_type());
    id<MTLComputePipelineState> pso = createPipeline(device, kernName, &error);
    TORCH_CHECK(pso, "Pipeline creation failed",
                error ? [NSString stringWithFormat:@": %@",
                                                   error.localizedDescription]
                            .UTF8String
                      : "");

    const uint32_t threads_per_tg =
        std::min<uint32_t>(512, hidden_size);

    dispatchNormKernel(
        pso, stream,
        getMTLBufferStorage(input),
        input.storage_offset() * input.element_size(),
        getMTLBufferStorage(residual),
        residual.storage_offset() * residual.element_size(),
        getMTLBufferStorage(weight),
        weight.storage_offset() * weight.element_size(),
        static_cast<float>(epsilon), static_cast<int32_t>(num_tokens),
        static_cast<int32_t>(hidden_size), input_stride,
        static_cast<uint32_t>(num_tokens), threads_per_tg);
  }
}
