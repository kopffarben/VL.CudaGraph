using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using ILGPU.Backends.EntryPoints;
using ILGPU.Backends.PTX;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using VL.Cuda.Core.Device;
using IlgpuContext = ILGPU.Context;

namespace VL.Cuda.Core.PTX.Compilation;

/// <summary>
/// Compiles C# methods to PTX via ILGPU's PTX backend.
/// Caches compiled kernels by method hash. No CUDA accelerator needed.
/// ~1-10ms per compilation.
/// </summary>
internal sealed class IlgpuCompiler : IDisposable
{
    private readonly IlgpuContext _ilgpuContext;
    private readonly PTXBackend _backend;
    private readonly DeviceContext _device;
    private readonly ConcurrentDictionary<string, LoadedKernel> _cache = new();
    private bool _disposed;

    public IlgpuCompiler(DeviceContext device)
    {
        _device = device ?? throw new ArgumentNullException(nameof(device));

        // Create ILGPU context without accelerator (pure compiler mode)
        _ilgpuContext = IlgpuContext.Create(builder => { });

        // Create PTX backend targeting the device's compute capability
        var arch = new CudaArchitecture(
            device.ComputeCapabilityMajor,
            device.ComputeCapabilityMinor);
        var isa = new CudaInstructionSet(
            device.ComputeCapabilityMajor,
            device.ComputeCapabilityMinor);

        _backend = new PTXBackend(_ilgpuContext, arch, isa, null);
    }

    /// <summary>
    /// Compile a C# method to PTX and load as a CUDA module. Cached by method hash.
    /// The method must follow ILGPU kernel conventions (Index1D first parameter, etc.).
    /// The descriptor is auto-expanded: each pointer (ArrayView) parameter becomes
    /// a (pointer, length) pair to match ILGPU's PTX layout.
    /// The returned LoadedKernel has the expanded descriptor.
    /// </summary>
    public LoadedKernel GetOrCompile(MethodInfo kernelMethod, KernelDescriptor descriptor)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var key = ComputeMethodHash(kernelMethod);
        if (_cache.TryGetValue(key, out var cached))
            return cached;

        var (ptxString, entryPointName) = CompileToStringWithName(kernelMethod);
        var ptxBytes = Encoding.ASCII.GetBytes(ptxString);

        // Expand descriptor: each pointer (ArrayView<T>) becomes (pointer, length) pair
        var expandedDescriptor = ExpandDescriptorForIlgpu(descriptor, entryPointName);

        var loaded = PtxLoader.LoadFromBytes(_device, ptxBytes, expandedDescriptor);
        _cache[key] = loaded;
        return loaded;
    }

    /// <summary>
    /// Compile a C# method to a PTX string without loading it.
    /// </summary>
    internal string CompileToString(MethodInfo kernelMethod)
    {
        return CompileToStringWithName(kernelMethod).PtxString;
    }

    /// <summary>
    /// Compile and return both the PTX string and the ILGPU-generated entry point name.
    /// </summary>
    internal (string PtxString, string EntryPointName) CompileToStringWithName(MethodInfo kernelMethod)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var entry = EntryPointDescription.FromImplicitlyGroupedKernel(kernelMethod);
        var compiled = _backend.Compile(entry, KernelSpecialization.Empty);

        if (compiled is PTXCompiledKernel ptxKernel)
            return (ptxKernel.PTXAssembly, compiled.Name);

        throw new InvalidOperationException(
            $"ILGPU compiled kernel is not PTXCompiledKernel: {compiled.GetType().Name}");
    }

    /// <summary>
    /// Remove a cached entry by method hash (SHA256).
    /// </summary>
    public bool Invalidate(string methodHash)
    {
        if (_cache.TryRemove(methodHash, out var removed))
        {
            removed.Dispose();
            return true;
        }
        return false;
    }

    public int CacheCount => _cache.Count;

    /// <summary>
    /// Expand a user-provided KernelDescriptor to match ILGPU's actual PTX parameter layout.
    /// ILGPU compiles implicitly-grouped kernels with:
    ///   param 0: kernel_length (.b32 for Index1D) — implicit, total element count
    ///   param 1..N: user params where each ArrayView&lt;T&gt; is a 16-byte struct {ptr, length}
    /// </summary>
    internal static KernelDescriptor ExpandDescriptorForIlgpu(KernelDescriptor original, string entryPointName)
    {
        var expandedParams = new List<KernelParamDescriptor>();

        // Param 0: implicit kernel_length (Index1D = int32, 4 bytes)
        expandedParams.Add(new KernelParamDescriptor
        {
            Name = "_kernel_length",
            Type = "int",
            IsPointer = false,
            Direction = ParamDirection.In,
            Index = 0,
            SizeBytes = 4,
        });

        // Params 1..N: user params (ArrayView<T> → 16-byte struct, scalars unchanged)
        for (int i = 0; i < original.Parameters.Count; i++)
        {
            var param = original.Parameters[i];
            expandedParams.Add(new KernelParamDescriptor
            {
                Name = param.Name,
                Type = param.Type,
                IsPointer = param.IsPointer,
                Direction = param.Direction,
                Index = expandedParams.Count,
                // ILGPU ArrayView<T> → 16-byte struct {void* ptr, long length}
                SizeBytes = param.IsPointer ? 16 : 0,
            });
        }

        return new KernelDescriptor
        {
            EntryPoint = entryPointName,
            Parameters = expandedParams,
            BlockSize = original.BlockSize,
            SharedMemoryBytes = original.SharedMemoryBytes,
        };
    }

    /// <summary>
    /// Compute the index remap from original descriptor indices to expanded (ILGPU PTX) indices.
    /// The implicit kernel_length param is inserted at index 0, so all user indices shift by 1.
    /// Used by KernelHandle.In()/Out() to translate user-facing indices to PTX indices.
    /// </summary>
    internal static int[] ComputeIndexRemap(KernelDescriptor original)
    {
        var remap = new int[original.Parameters.Count];
        for (int i = 0; i < original.Parameters.Count; i++)
        {
            remap[i] = i + 1; // shift by 1 for the implicit kernel_length at index 0
        }
        return remap;
    }

    /// <summary>
    /// Compute a SHA256 hash from a MethodInfo, matching BlockBuilder usage.
    /// </summary>
    internal static string ComputeMethodHash(MethodInfo method)
    {
        var key = $"{method.DeclaringType?.AssemblyQualifiedName}.{method.Name}.{method.MetadataToken}";
        var hash = SHA256.HashData(Encoding.UTF8.GetBytes(key));
        return Convert.ToHexString(hash);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        foreach (var entry in _cache.Values)
            entry.Dispose();
        _cache.Clear();

        _backend.Dispose();
        _ilgpuContext.Dispose();
    }
}
