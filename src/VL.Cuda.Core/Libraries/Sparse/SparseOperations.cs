using System;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaSparse;
using VL.Cuda.Core.Blocks.Builder;
using VL.Cuda.Core.Graph;

namespace VL.Cuda.Core.Libraries.Sparse;

/// <summary>
/// High-level cuSPARSE wrappers that produce CapturedHandle entries for BlockBuilder.
/// Sparse matrix operations via stream capture.
/// Note: cuSPARSE operations require workspace buffers that must be pre-allocated.
/// </summary>
public static class SparseOperations
{
    /// <summary>
    /// Sparse matrix-vector multiply (SpMV): y = alpha*A*x + beta*y.
    /// A is sparse CSR (M x N, nnz non-zeros), x and y are dense vectors.
    /// Buffers: In[0]=csrValues, In[1]=csrRowOffsets, In[2]=csrColInd, In[3]=x,
    ///          Out[0]=y, Out[1]=workspace.
    /// </summary>
    public static CapturedHandle SpMV(
        BlockBuilder builder,
        LibraryHandleCache libs,
        int m, int n, int nnz,
        int workspaceSizeBytes)
    {
        var descriptor = new CapturedNodeDescriptor("cuSPARSE.SpMV",
            inputs: new[]
            {
                CapturedParam.Pointer("csrValues", "float*"),
                CapturedParam.Pointer("csrRowOffsets", "int*"),
                CapturedParam.Pointer("csrColInd", "int*"),
                CapturedParam.Pointer("x", "float*"),
            },
            outputs: new[]
            {
                CapturedParam.Pointer("y", "float*"),
                CapturedParam.Pointer("Workspace", "byte*"),
            });

        return builder.AddCaptured((stream, buffers) =>
        {
            var sparse = libs.GetOrCreateSparse();
            sparse.SetStream(stream);

            // Create temporary wrappers around raw pointers for the managed API
            var csrValues = new CudaDeviceVariable<float>(buffers[0], (SizeT)(nnz * sizeof(float)));
            var csrRowOffsets = new CudaDeviceVariable<int>(buffers[1], (SizeT)((m + 1) * sizeof(int)));
            var csrColInd = new CudaDeviceVariable<int>(buffers[2], (SizeT)(nnz * sizeof(int)));
            var vecX = new CudaDeviceVariable<float>(buffers[3], (SizeT)(n * sizeof(float)));
            var vecY = new CudaDeviceVariable<float>(buffers[4], (SizeT)(m * sizeof(float)));
            var workspace = new CudaDeviceVariable<byte>(buffers[5], (SizeT)workspaceSizeBytes);

            using var matA = ConstSparseMatrix<int, float>.CreateConstCSR(m, n, nnz,
                csrRowOffsets, csrColInd, csrValues, IndexBase.Zero);
            using var denseX = new ConstDenseVector<float>(n, vecX);
            using var denseY = new DenseVector<float>(m, vecY);

            float alpha = 1.0f;
            float beta = 0.0f;
            sparse.MV<int, float, float>(cusparseOperation.NonTranspose, alpha,
                matA, denseX, beta, denseY,
                cudaDataType.CUDA_R_32F, SpMVAlg.CUSPARSE_SPMV_ALG_DEFAULT, workspace);
        }, descriptor);
    }
}
