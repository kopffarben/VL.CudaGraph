using System;
using System.Collections.Generic;
using System.Linq;
using ILGPU;
using ILGPU.Runtime;
using VL.Cuda.Core.Device;
using VL.Cuda.Core.PTX;
using VL.Cuda.Core.PTX.Compilation;
using Xunit;

namespace VL.Cuda.Tests.Compilation;

public class IlgpuCompilerTests : IDisposable
{
    private readonly DeviceContext _device = new(0);

    /// <summary>
    /// Simple ILGPU kernel: scales each element by 2.
    /// Uses Index1D (implicitly grouped) convention.
    /// </summary>
    static void ScaleKernel(Index1D i, ArrayView<float> data, ArrayView<float> output)
    {
        output[i] = data[i] * 2.0f;
    }

    /// <summary>
    /// Another kernel with different signature for cache tests.
    /// </summary>
    static void AddKernel(Index1D i, ArrayView<float> a, ArrayView<float> b, ArrayView<float> c)
    {
        c[i] = a[i] + b[i];
    }

    private static KernelDescriptor MakeDescriptor(string entryPoint, params (string Name, string Type, bool IsPointer)[] parameters)
    {
        var paramDescs = new List<KernelParamDescriptor>();
        for (int idx = 0; idx < parameters.Length; idx++)
        {
            var (name, type, isPointer) = parameters[idx];
            paramDescs.Add(new KernelParamDescriptor
            {
                Name = name,
                Type = type,
                IsPointer = isPointer,
                Direction = isPointer ? ParamDirection.InOut : ParamDirection.In,
                Index = idx,
            });
        }
        return new KernelDescriptor
        {
            EntryPoint = entryPoint,
            Parameters = paramDescs,
            BlockSize = 256,
        };
    }

    [Fact]
    public void CompileToString_ProducesPTX()
    {
        using var compiler = new IlgpuCompiler(_device);

        var method = typeof(IlgpuCompilerTests).GetMethod(nameof(ScaleKernel),
            System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)!;

        var ptx = compiler.CompileToString(method);

        Assert.NotNull(ptx);
        Assert.NotEmpty(ptx);
        Assert.Contains(".version", ptx);
        Assert.Contains(".target", ptx);
    }

    [Fact]
    public void GetOrCompile_LoadsModule()
    {
        using var compiler = new IlgpuCompiler(_device);

        var method = typeof(IlgpuCompilerTests).GetMethod(nameof(ScaleKernel),
            System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)!;

        var descriptor = MakeDescriptor("ScaleKernel",
            ("data", "float*", true), ("output", "float*", true));

        var loaded = compiler.GetOrCompile(method, descriptor);

        Assert.NotNull(loaded);
        Assert.Contains("ScaleKernel", loaded.Descriptor.EntryPoint);
        // Expanded: (_kernel_length, data, output) = 3 params
        Assert.Equal(3, loaded.Descriptor.Parameters.Count);
        Assert.False(loaded.Descriptor.Parameters[0].IsPointer);  // _kernel_length
        Assert.Equal(4, loaded.Descriptor.Parameters[0].SizeBytes); // int32
        Assert.True(loaded.Descriptor.Parameters[1].IsPointer);   // data (ArrayView struct)
        Assert.Equal(16, loaded.Descriptor.Parameters[1].SizeBytes); // 16-byte struct
        Assert.True(loaded.Descriptor.Parameters[2].IsPointer);   // output (ArrayView struct)
        Assert.Equal(16, loaded.Descriptor.Parameters[2].SizeBytes);
    }

    [Fact]
    public void GetOrCompile_CacheHit_ReturnsSameInstance()
    {
        using var compiler = new IlgpuCompiler(_device);

        var method = typeof(IlgpuCompilerTests).GetMethod(nameof(ScaleKernel),
            System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)!;

        var descriptor = MakeDescriptor("ScaleKernel",
            ("data", "float*", true), ("output", "float*", true));

        var first = compiler.GetOrCompile(method, descriptor);
        var second = compiler.GetOrCompile(method, descriptor);

        Assert.Same(first, second);
        Assert.Equal(1, compiler.CacheCount);
    }

    [Fact]
    public void GetOrCompile_DifferentMethods_DifferentCacheEntries()
    {
        using var compiler = new IlgpuCompiler(_device);

        var scaleMethod = typeof(IlgpuCompilerTests).GetMethod(nameof(ScaleKernel),
            System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)!;
        var addMethod = typeof(IlgpuCompilerTests).GetMethod(nameof(AddKernel),
            System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)!;

        var descScale = MakeDescriptor("ScaleKernel",
            ("data", "float*", true), ("output", "float*", true));
        var descAdd = MakeDescriptor("AddKernel",
            ("a", "float*", true), ("b", "float*", true), ("c", "float*", true));

        var scaleLoaded = compiler.GetOrCompile(scaleMethod, descScale);
        var addLoaded = compiler.GetOrCompile(addMethod, descAdd);

        Assert.NotSame(scaleLoaded, addLoaded);
        Assert.Equal(2, compiler.CacheCount);
    }

    [Fact]
    public void Invalidate_RemovesFromCache()
    {
        using var compiler = new IlgpuCompiler(_device);

        var method = typeof(IlgpuCompilerTests).GetMethod(nameof(ScaleKernel),
            System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)!;

        var descriptor = MakeDescriptor("ScaleKernel",
            ("data", "float*", true), ("output", "float*", true));

        compiler.GetOrCompile(method, descriptor);
        Assert.Equal(1, compiler.CacheCount);

        var hash = IlgpuCompiler.ComputeMethodHash(method);
        var invalidated = compiler.Invalidate(hash);

        Assert.True(invalidated);
        Assert.Equal(0, compiler.CacheCount);
    }

    [Fact]
    public void Invalidate_NonExistent_ReturnsFalse()
    {
        using var compiler = new IlgpuCompiler(_device);
        Assert.False(compiler.Invalidate("nonexistent_hash"));
    }

    [Fact]
    public void ExpandDescriptor_InsertsKernelLength()
    {
        var descriptor = MakeDescriptor("Test",
            ("data", "float*", true), ("output", "float*", true));

        var expanded = IlgpuCompiler.ExpandDescriptorForIlgpu(descriptor, "Kernel_Test");

        // kernel_length + data (ArrayView struct) + output (ArrayView struct) = 3 params
        Assert.Equal(3, expanded.Parameters.Count);
        Assert.Equal("Kernel_Test", expanded.EntryPoint);

        // Param 0: implicit kernel_length (.b32, 4 bytes)
        Assert.Equal("_kernel_length", expanded.Parameters[0].Name);
        Assert.False(expanded.Parameters[0].IsPointer);
        Assert.Equal(4, expanded.Parameters[0].SizeBytes);

        // Param 1: data (ArrayView struct, 16 bytes)
        Assert.Equal("data", expanded.Parameters[1].Name);
        Assert.True(expanded.Parameters[1].IsPointer);
        Assert.Equal(16, expanded.Parameters[1].SizeBytes);

        // Param 2: output (ArrayView struct, 16 bytes)
        Assert.Equal("output", expanded.Parameters[2].Name);
        Assert.True(expanded.Parameters[2].IsPointer);
        Assert.Equal(16, expanded.Parameters[2].SizeBytes);
    }

    [Fact]
    public void ExpandDescriptor_MixedPointerAndScalar()
    {
        var descriptor = MakeDescriptor("Test",
            ("data", "float*", true), ("scale", "float", false), ("output", "float*", true));

        var expanded = IlgpuCompiler.ExpandDescriptorForIlgpu(descriptor, "K");

        // kernel_length + data (ArrayView struct) + scale + output (ArrayView struct) = 4 params
        Assert.Equal(4, expanded.Parameters.Count);
        Assert.Equal("_kernel_length", expanded.Parameters[0].Name);
        Assert.Equal("data", expanded.Parameters[1].Name);
        Assert.True(expanded.Parameters[1].IsPointer);
        Assert.Equal(16, expanded.Parameters[1].SizeBytes);
        Assert.Equal("scale", expanded.Parameters[2].Name);
        Assert.False(expanded.Parameters[2].IsPointer);
        Assert.Equal("output", expanded.Parameters[3].Name);
        Assert.True(expanded.Parameters[3].IsPointer);
        Assert.Equal(16, expanded.Parameters[3].SizeBytes);
    }

    [Fact]
    public void ComputeIndexRemap_PointerOnlyParams()
    {
        var descriptor = MakeDescriptor("Test",
            ("data", "float*", true), ("output", "float*", true));

        var remap = IlgpuCompiler.ComputeIndexRemap(descriptor);

        Assert.Equal(2, remap.Length);
        Assert.Equal(1, remap[0]); // data → expanded index 1 (shift +1 for kernel_length)
        Assert.Equal(2, remap[1]); // output → expanded index 2
    }

    [Fact]
    public void ComputeIndexRemap_MixedParams()
    {
        var descriptor = MakeDescriptor("Test",
            ("data", "float*", true), ("scale", "float", false), ("output", "float*", true));

        var remap = IlgpuCompiler.ComputeIndexRemap(descriptor);

        Assert.Equal(3, remap.Length);
        Assert.Equal(1, remap[0]); // data → expanded index 1 (shift +1 for kernel_length)
        Assert.Equal(2, remap[1]); // scale → expanded index 2
        Assert.Equal(3, remap[2]); // output → expanded index 3
    }

    [Fact]
    public void CompileToString_HasCorrectParamLayout()
    {
        using var compiler = new IlgpuCompiler(_device);

        var method = typeof(IlgpuCompilerTests).GetMethod(nameof(ScaleKernel),
            System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)!;

        var ptx = compiler.CompileToString(method);

        // Count .param entries in the PTX
        var paramLines = ptx.Split('\n')
            .Where(l => l.Trim().StartsWith(".param"))
            .ToList();

        // ILGPU layout for ScaleKernel(Index1D, ArrayView<float>, ArrayView<float>):
        // param 0: .b32 (Index1D / kernel_length)
        // param 1: .b8[16] (ArrayView struct = {ptr, length})
        // param 2: .b8[16] (ArrayView struct = {ptr, length})
        Assert.Equal(3, paramLines.Count);
        Assert.Contains(".b32", paramLines[0]);     // kernel_length
        Assert.Contains("[16]", paramLines[1]);     // data ArrayView struct
        Assert.Contains("[16]", paramLines[2]);     // output ArrayView struct
    }

    [Fact]
    public void ComputeMethodHash_Deterministic()
    {
        var method = typeof(IlgpuCompilerTests).GetMethod(nameof(ScaleKernel),
            System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)!;

        var hash1 = IlgpuCompiler.ComputeMethodHash(method);
        var hash2 = IlgpuCompiler.ComputeMethodHash(method);

        Assert.Equal(hash1, hash2);
        Assert.Equal(64, hash1.Length); // SHA256 = 32 bytes = 64 hex chars
    }

    public void Dispose()
    {
        _device.Dispose();
    }
}
