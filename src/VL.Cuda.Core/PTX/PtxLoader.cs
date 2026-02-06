using System;
using System.IO;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using VL.Cuda.Core.Device;

namespace VL.Cuda.Core.PTX;

/// <summary>
/// Loads .ptx files and their companion .json metadata.
/// Returns a loaded module + kernel descriptor pair.
/// </summary>
public static class PtxLoader
{
    /// <summary>
    /// Load a PTX file and its companion JSON metadata.
    /// JSON file is expected at the same path with .json extension.
    /// </summary>
    public static LoadedKernel Load(DeviceContext device, string ptxPath)
    {
        if (!File.Exists(ptxPath))
            throw new FileNotFoundException($"PTX file not found: {ptxPath}");

        var jsonPath = Path.ChangeExtension(ptxPath, ".json");
        if (!File.Exists(jsonPath))
            throw new FileNotFoundException($"Metadata JSON not found: {jsonPath}");

        var descriptor = PtxMetadata.Parse(jsonPath);
        var module = device.Context.LoadModulePTX(ptxPath);
        var kernel = new CudaKernel(descriptor.EntryPoint, module);

        if (descriptor.BlockSize > 0)
            kernel.BlockDimensions = new dim3(descriptor.BlockSize, 1, 1);

        if (descriptor.SharedMemoryBytes > 0)
            kernel.DynamicSharedMemory = (uint)descriptor.SharedMemoryBytes;

        return new LoadedKernel(device, module, kernel, descriptor);
    }

    /// <summary>
    /// Load a PTX file from bytes with a pre-parsed descriptor.
    /// </summary>
    public static LoadedKernel LoadFromBytes(DeviceContext device, byte[] ptxBytes, KernelDescriptor descriptor)
    {
        var module = device.Context.LoadModulePTX(ptxBytes);
        var kernel = new CudaKernel(descriptor.EntryPoint, module);

        if (descriptor.BlockSize > 0)
            kernel.BlockDimensions = new dim3(descriptor.BlockSize, 1, 1);

        if (descriptor.SharedMemoryBytes > 0)
            kernel.DynamicSharedMemory = (uint)descriptor.SharedMemoryBytes;

        return new LoadedKernel(device, module, kernel, descriptor);
    }
}

/// <summary>
/// A loaded CUDA module with its kernel handle and metadata.
/// </summary>
public sealed class LoadedKernel : IDisposable
{
    private readonly DeviceContext _device;

    public CUmodule Module { get; }
    public CudaKernel Kernel { get; }
    public KernelDescriptor Descriptor { get; }

    internal LoadedKernel(DeviceContext device, CUmodule module, CudaKernel kernel, KernelDescriptor descriptor)
    {
        _device = device;
        Module = module;
        Kernel = kernel;
        Descriptor = descriptor;
    }

    public void Dispose()
    {
        _device.Context.UnloadModule(Module);
    }
}
