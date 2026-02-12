using System;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaBlas;
using VL.Cuda.Core.Blocks.Builder;
using VL.Cuda.Core.Graph;

namespace VL.Cuda.Core.Libraries.Blas;

/// <summary>
/// High-level cuBLAS wrappers that produce CapturedHandle entries for BlockBuilder.
/// Each operation wraps a stream-captured cuBLAS call as a CapturedNode.
/// Buffer bindings follow descriptor order: [inputs..., outputs..., scalars...].
/// </summary>
public static class BlasOperations
{
    /// <summary>
    /// Single-precision matrix multiply: C = alpha*A*B + beta*C.
    /// A is (M x K), B is (K x N), C is (M x N). Column-major layout.
    /// Buffers: In[0]=A, In[1]=B, Out[0]=C.
    /// </summary>
    public static CapturedHandle Sgemm(
        BlockBuilder builder,
        LibraryHandleCache libs,
        int m, int n, int k,
        float alpha = 1.0f, float beta = 0.0f,
        Operation transA = Operation.NonTranspose,
        Operation transB = Operation.NonTranspose)
    {
        var descriptor = new CapturedNodeDescriptor("cuBLAS.Sgemm",
            inputs: new[] { CapturedParam.Pointer("A", "float*"), CapturedParam.Pointer("B", "float*") },
            outputs: new[] { CapturedParam.Pointer("C", "float*") });

        return builder.AddCaptured((stream, buffers) =>
        {
            // buffers layout: [A, B, C] (inputs first, then outputs)
            var blas = libs.GetOrCreateBlas();
            blas.Stream = stream;
            blas.PointerMode = PointerMode.Host;

            int lda = transA == Operation.NonTranspose ? m : k;
            int ldb = transB == Operation.NonTranspose ? k : n;
            int ldc = m;

            CudaBlasNativeMethods.cublasSgemm_v2(
                blas.CublasHandle, transA, transB,
                m, n, k,
                ref alpha,
                buffers[0], lda,  // A
                buffers[1], ldb,  // B
                ref beta,
                buffers[2], ldc   // C
            );
        }, descriptor);
    }

    /// <summary>
    /// Double-precision matrix multiply: C = alpha*A*B + beta*C.
    /// Buffers: In[0]=A, In[1]=B, Out[0]=C.
    /// </summary>
    public static CapturedHandle Dgemm(
        BlockBuilder builder,
        LibraryHandleCache libs,
        int m, int n, int k,
        double alpha = 1.0, double beta = 0.0,
        Operation transA = Operation.NonTranspose,
        Operation transB = Operation.NonTranspose)
    {
        var descriptor = new CapturedNodeDescriptor("cuBLAS.Dgemm",
            inputs: new[] { CapturedParam.Pointer("A", "double*"), CapturedParam.Pointer("B", "double*") },
            outputs: new[] { CapturedParam.Pointer("C", "double*") });

        return builder.AddCaptured((stream, buffers) =>
        {
            var blas = libs.GetOrCreateBlas();
            blas.Stream = stream;
            blas.PointerMode = PointerMode.Host;

            int lda = transA == Operation.NonTranspose ? m : k;
            int ldb = transB == Operation.NonTranspose ? k : n;
            int ldc = m;

            CudaBlasNativeMethods.cublasDgemm_v2(
                blas.CublasHandle, transA, transB,
                m, n, k,
                ref alpha,
                buffers[0], lda,
                buffers[1], ldb,
                ref beta,
                buffers[2], ldc
            );
        }, descriptor);
    }

    /// <summary>
    /// Single-precision matrix-vector multiply: y = alpha*A*x + beta*y.
    /// A is (M x N), x is (N), y is (M).
    /// Buffers: In[0]=A, In[1]=x, Out[0]=y.
    /// </summary>
    public static CapturedHandle Sgemv(
        BlockBuilder builder,
        LibraryHandleCache libs,
        int m, int n,
        float alpha = 1.0f, float beta = 0.0f,
        Operation transA = Operation.NonTranspose)
    {
        var descriptor = new CapturedNodeDescriptor("cuBLAS.Sgemv",
            inputs: new[] { CapturedParam.Pointer("A", "float*"), CapturedParam.Pointer("x", "float*") },
            outputs: new[] { CapturedParam.Pointer("y", "float*") });

        return builder.AddCaptured((stream, buffers) =>
        {
            var blas = libs.GetOrCreateBlas();
            blas.Stream = stream;
            blas.PointerMode = PointerMode.Host;

            CudaBlasNativeMethods.cublasSgemv_v2(
                blas.CublasHandle, transA,
                m, n,
                ref alpha,
                buffers[0], m,  // A
                buffers[1], 1,  // x
                ref beta,
                buffers[2], 1   // y
            );
        }, descriptor);
    }

    /// <summary>
    /// Single-precision vector scaling: x = alpha * x.
    /// Buffers: Out[0]=x (in-place).
    /// </summary>
    public static CapturedHandle Sscal(
        BlockBuilder builder,
        LibraryHandleCache libs,
        int n,
        float alpha = 1.0f)
    {
        var descriptor = new CapturedNodeDescriptor("cuBLAS.Sscal",
            outputs: new[] { CapturedParam.Pointer("x", "float*") });

        return builder.AddCaptured((stream, buffers) =>
        {
            var blas = libs.GetOrCreateBlas();
            blas.Stream = stream;
            blas.PointerMode = PointerMode.Host;

            CudaBlasNativeMethods.cublasSscal_v2(
                blas.CublasHandle, n,
                ref alpha,
                buffers[0], 1   // x (output at index 0, no inputs)
            );
        }, descriptor);
    }
}
