using System;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaRand;
using VL.Cuda.Core.Blocks.Builder;
using VL.Cuda.Core.Graph;

namespace VL.Cuda.Core.Libraries.Rand;

/// <summary>
/// High-level cuRAND wrappers that produce CapturedHandle entries for BlockBuilder.
/// Random number generation via stream capture.
/// </summary>
public static class RandOperations
{
    /// <summary>
    /// Generate uniform random float values in [0, 1).
    /// Buffers: Out[0]=output.
    /// </summary>
    public static CapturedHandle GenerateUniform(
        BlockBuilder builder,
        LibraryHandleCache libs,
        int count)
    {
        var descriptor = new CapturedNodeDescriptor("cuRAND.Uniform",
            outputs: new[] { CapturedParam.Pointer("Output", "float*") });

        return builder.AddCaptured((stream, buffers) =>
        {
            var rand = libs.GetOrCreateRand();
            rand.SetStream(stream);
            rand.GenerateUniform32(buffers[0], (SizeT)count);
        }, descriptor);
    }

    /// <summary>
    /// Generate normal-distributed random float values.
    /// Buffers: Out[0]=output.
    /// </summary>
    public static CapturedHandle GenerateNormal(
        BlockBuilder builder,
        LibraryHandleCache libs,
        int count,
        float mean = 0.0f,
        float stddev = 1.0f)
    {
        var descriptor = new CapturedNodeDescriptor("cuRAND.Normal",
            outputs: new[] { CapturedParam.Pointer("Output", "float*") });

        return builder.AddCaptured((stream, buffers) =>
        {
            var rand = libs.GetOrCreateRand();
            rand.SetStream(stream);
            rand.GenerateNormal32(buffers[0], (SizeT)count, mean, stddev);
        }, descriptor);
    }
}
