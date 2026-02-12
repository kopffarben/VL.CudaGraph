using System;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaBlas;
using ManagedCuda.CudaSolve;
using VL.Cuda.Core.Blocks.Builder;
using VL.Cuda.Core.Graph;

namespace VL.Cuda.Core.Libraries.Solve;

/// <summary>
/// High-level cuSOLVER wrappers that produce CapturedHandle entries for BlockBuilder.
/// Dense linear algebra solvers via stream capture.
/// </summary>
public static class SolveOperations
{
    /// <summary>
    /// LU factorization of an M x N matrix: A = P * L * U.
    /// Buffers: Out[0]=A (in-place LU), Out[1]=workspace, Out[2]=devIpiv, Out[3]=devInfo.
    /// </summary>
    public static CapturedHandle Sgetrf(
        BlockBuilder builder,
        LibraryHandleCache libs,
        int m, int n)
    {
        var descriptor = new CapturedNodeDescriptor("cuSOLVER.Sgetrf",
            outputs: new[]
            {
                CapturedParam.Pointer("A", "float*"),
                CapturedParam.Pointer("Workspace", "float*"),
                CapturedParam.Pointer("Ipiv", "int*"),
                CapturedParam.Pointer("Info", "int*"),
            });

        return builder.AddCaptured((stream, buffers) =>
        {
            var solver = libs.GetOrCreateSolveDense();
            solver.SetStream(new CudaStream(stream));

            // Create CudaDeviceVariable wrappers around raw pointers
            var matA = new CudaDeviceVariable<float>(buffers[0], (SizeT)(m * n * sizeof(float)));
            var workspace = new CudaDeviceVariable<float>(buffers[1], (SizeT)(m * n * sizeof(float)));
            var devIpiv = new CudaDeviceVariable<int>(buffers[2], (SizeT)(System.Math.Min(m, n) * sizeof(int)));
            var devInfo = new CudaDeviceVariable<int>(buffers[3], (SizeT)sizeof(int));

            solver.Getrf(m, n, matA, m, workspace, devIpiv, devInfo);
        }, descriptor);
    }

    /// <summary>
    /// Solve a linear system A*X = B using LU factorization.
    /// A must have been factored by Sgetrf first.
    /// Buffers: In[0]=A (LU), In[1]=Ipiv, Out[0]=B (in-place solution X), Out[1]=Info.
    /// </summary>
    public static CapturedHandle Sgetrs(
        BlockBuilder builder,
        LibraryHandleCache libs,
        int n, int nrhs,
        Operation trans = Operation.NonTranspose)
    {
        var descriptor = new CapturedNodeDescriptor("cuSOLVER.Sgetrs",
            inputs: new[]
            {
                CapturedParam.Pointer("A_LU", "float*"),
                CapturedParam.Pointer("Ipiv", "int*"),
            },
            outputs: new[]
            {
                CapturedParam.Pointer("B", "float*"),
                CapturedParam.Pointer("Info", "int*"),
            });

        return builder.AddCaptured((stream, buffers) =>
        {
            var solver = libs.GetOrCreateSolveDense();
            solver.SetStream(new CudaStream(stream));

            // Create CudaDeviceVariable wrappers around raw pointers
            var matA = new CudaDeviceVariable<float>(buffers[0], (SizeT)(n * n * sizeof(float)));
            var devIpiv = new CudaDeviceVariable<int>(buffers[1], (SizeT)(n * sizeof(int)));
            var matB = new CudaDeviceVariable<float>(buffers[2], (SizeT)(n * nrhs * sizeof(float)));
            var devInfo = new CudaDeviceVariable<int>(buffers[3], (SizeT)sizeof(int));

            // Operation enum from ManagedCuda.CudaBlas is used directly by cuSOLVER
            solver.Getrs(trans, n, nrhs, matA, n, devIpiv, matB, n, devInfo);
        }, descriptor);
    }
}
