using System;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaFFT;
using VL.Cuda.Core.Blocks.Builder;
using VL.Cuda.Core.Graph;

namespace VL.Cuda.Core.Libraries.FFT;

/// <summary>
/// High-level cuFFT wrappers that produce CapturedHandle entries for BlockBuilder.
/// Each operation wraps a stream-captured cuFFT plan execution as a CapturedNode.
/// </summary>
public static class FftOperations
{
    /// <summary>
    /// 1D FFT Forward transform (out-of-place).
    /// Buffers: In[0]=input, Out[0]=output.
    /// </summary>
    public static CapturedHandle Forward1D(
        BlockBuilder builder,
        LibraryHandleCache libs,
        int nx,
        cufftType type = cufftType.C2C,
        int batch = 1)
    {
        var descriptor = new CapturedNodeDescriptor("cuFFT.Forward1D",
            inputs: new[] { CapturedParam.Pointer("Input", "float2*") },
            outputs: new[] { CapturedParam.Pointer("Output", "float2*") });

        return builder.AddCaptured((stream, buffers) =>
        {
            var plan = libs.GetOrCreateFFT1D(nx, type, batch);
            CudaFFTNativeMethods.cufftSetStream(plan.Handle, stream);
            plan.Exec(buffers[0], buffers[1], TransformDirection.Forward);
        }, descriptor);
    }

    /// <summary>
    /// 1D FFT Inverse transform (out-of-place).
    /// Buffers: In[0]=input, Out[0]=output.
    /// </summary>
    public static CapturedHandle Inverse1D(
        BlockBuilder builder,
        LibraryHandleCache libs,
        int nx,
        cufftType type = cufftType.C2C,
        int batch = 1)
    {
        var descriptor = new CapturedNodeDescriptor("cuFFT.Inverse1D",
            inputs: new[] { CapturedParam.Pointer("Input", "float2*") },
            outputs: new[] { CapturedParam.Pointer("Output", "float2*") });

        return builder.AddCaptured((stream, buffers) =>
        {
            var plan = libs.GetOrCreateFFT1D(nx, type, batch);
            CudaFFTNativeMethods.cufftSetStream(plan.Handle, stream);
            plan.Exec(buffers[0], buffers[1], TransformDirection.Inverse);
        }, descriptor);
    }

    /// <summary>
    /// 1D Real-to-Complex FFT (out-of-place).
    /// Input: N real values, Output: N/2+1 complex values.
    /// Buffers: In[0]=input (float*), Out[0]=output (float2*).
    /// </summary>
    public static CapturedHandle R2C1D(
        BlockBuilder builder,
        LibraryHandleCache libs,
        int nx,
        int batch = 1)
    {
        var descriptor = new CapturedNodeDescriptor("cuFFT.R2C1D",
            inputs: new[] { CapturedParam.Pointer("Input", "float*") },
            outputs: new[] { CapturedParam.Pointer("Output", "float2*") });

        return builder.AddCaptured((stream, buffers) =>
        {
            var plan = libs.GetOrCreateFFT1D(nx, cufftType.R2C, batch);
            CudaFFTNativeMethods.cufftSetStream(plan.Handle, stream);
            plan.Exec(buffers[0], buffers[1]);
        }, descriptor);
    }

    /// <summary>
    /// 1D Complex-to-Real inverse FFT (out-of-place).
    /// Input: N/2+1 complex values, Output: N real values.
    /// Buffers: In[0]=input (float2*), Out[0]=output (float*).
    /// </summary>
    public static CapturedHandle C2R1D(
        BlockBuilder builder,
        LibraryHandleCache libs,
        int nx,
        int batch = 1)
    {
        var descriptor = new CapturedNodeDescriptor("cuFFT.C2R1D",
            inputs: new[] { CapturedParam.Pointer("Input", "float2*") },
            outputs: new[] { CapturedParam.Pointer("Output", "float*") });

        return builder.AddCaptured((stream, buffers) =>
        {
            var plan = libs.GetOrCreateFFT1D(nx, cufftType.C2R, batch);
            CudaFFTNativeMethods.cufftSetStream(plan.Handle, stream);
            plan.Exec(buffers[0], buffers[1]);
        }, descriptor);
    }
}
