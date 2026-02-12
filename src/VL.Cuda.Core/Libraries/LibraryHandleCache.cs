using System;
using System.Collections.Generic;
using ManagedCuda.CudaBlas;
using ManagedCuda.CudaFFT;
using ManagedCuda.CudaSparse;
using ManagedCuda.CudaRand;
using ManagedCuda.CudaSolve;

namespace VL.Cuda.Core.Libraries;

/// <summary>
/// Lazy-initialized cache for CUDA library handles. One instance per CudaContext.
/// Handles are expensive to create, so we cache them and reuse across captures.
/// Each handle is created on first access and disposed when the cache is disposed.
/// </summary>
public sealed class LibraryHandleCache : IDisposable
{
    private CudaBlas? _blas;
    private CudaSparseContext? _sparse;
    private CudaRandDevice? _rand;
    private CudaSolveDense? _solveDense;
    private readonly Dictionary<(int Nx, cufftType Type, int Batch), CudaFFTPlan1D> _fftPlans = new();
    private bool _disposed;

    /// <summary>
    /// Get or create the cuBLAS handle.
    /// </summary>
    public CudaBlas GetOrCreateBlas()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _blas ??= new CudaBlas();
    }

    /// <summary>
    /// Get or create a cuFFT 1D plan with the given configuration.
    /// Plans are cached by (nx, type, batch) since they are configuration-dependent.
    /// </summary>
    public CudaFFTPlan1D GetOrCreateFFT1D(int nx, cufftType type, int batch = 1)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var key = (nx, type, batch);
        if (!_fftPlans.TryGetValue(key, out var plan))
        {
            plan = new CudaFFTPlan1D(nx, type, batch);
            _fftPlans[key] = plan;
        }
        return plan;
    }

    /// <summary>
    /// Get or create the cuSPARSE context.
    /// </summary>
    public CudaSparseContext GetOrCreateSparse()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _sparse ??= new CudaSparseContext();
    }

    /// <summary>
    /// Get or create the cuRAND device generator.
    /// </summary>
    public CudaRandDevice GetOrCreateRand(GeneratorType type = GeneratorType.PseudoDefault)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _rand ??= new CudaRandDevice(type);
    }

    /// <summary>
    /// Get or create the cuSOLVER dense handle.
    /// </summary>
    public CudaSolveDense GetOrCreateSolveDense()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _solveDense ??= new CudaSolveDense();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _blas?.Dispose();
        _blas = null;

        foreach (var plan in _fftPlans.Values)
            plan.Dispose();
        _fftPlans.Clear();

        _sparse?.Dispose();
        _sparse = null;

        _rand?.Dispose();
        _rand = null;

        _solveDense?.Dispose();
        _solveDense = null;
    }
}
