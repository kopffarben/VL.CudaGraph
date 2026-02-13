using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ILGPU;
using ILGPU.Runtime;
using VL.Cuda.Core.Blocks;
using VL.Cuda.Core.Blocks.Builder;
using VL.Cuda.Core.Buffers;
using VL.Cuda.Core.Context;
using VL.Cuda.Core.Device;
using VL.Cuda.Core.Engine;
using VL.Cuda.Core.PTX;
using VL.Cuda.Tests.Helpers;
using Xunit;

namespace VL.Cuda.Tests.Integration;

public class PatchableKernelTests : IDisposable
{
    private readonly DeviceContext _device = new(0);
    private readonly CudaEngine _engine;

    private static string ScalarMulPath => Path.Combine(AppContext.BaseDirectory, "TestKernels", "scalar_mul.ptx");

    public PatchableKernelTests()
    {
        _engine = new CudaEngine(new CudaContext(_device));
    }

    // --- ILGPU Kernel Methods ---

    /// <summary>
    /// ILGPU kernel: scale each element by 2.
    /// Uses Index1D (implicitly grouped) convention.
    /// </summary>
    static void ScaleBy2Kernel(Index1D i, ArrayView<float> data, ArrayView<float> output)
    {
        output[i] = data[i] * 2.0f;
    }

    // --- NVRTC CUDA C++ Sources ---

    private const string NvrtcScaleSource = @"
extern ""C"" __global__ void nvrtc_scale(float* data, float* output, float scale, unsigned int N)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        output[idx] = data[idx] * scale;
}
";

    // --- Helper ---

    private static KernelDescriptor MakeDescriptor(string entryPoint, params (string Name, string Type, bool IsPointer, ParamDirection Dir)[] parameters)
    {
        var paramDescs = new List<KernelParamDescriptor>();
        for (int idx = 0; idx < parameters.Length; idx++)
        {
            var (name, type, isPointer, dir) = parameters[idx];
            paramDescs.Add(new KernelParamDescriptor
            {
                Name = name,
                Type = type,
                IsPointer = isPointer,
                Direction = dir,
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

    // --- ILGPU Tests ---

    [Fact]
    public void IlgpuKernel_CompilesAndExecutes()
    {
        const int N = 256;

        using var bufIn = GpuBuffer<float>.Allocate(_device, N);
        using var bufOut = GpuBuffer<float>.Allocate(_device, N);

        var dataIn = Enumerable.Range(0, N).Select(i => (float)i).ToArray();
        bufIn.Upload(dataIn);

        var scaleMethod = typeof(PatchableKernelTests).GetMethod(nameof(ScaleBy2Kernel),
            System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)!;

        // Entry point name will be overridden by ILGPU's generated name (Kernel_ScaleBy2Kernel)
        var descriptor = MakeDescriptor("ScaleBy2Kernel",
            ("data", "float*", true, ParamDirection.In),
            ("output", "float*", true, ParamDirection.Out));

        var block = new BuiltTestBlock(_engine.Context, ScalarMulPath, "IlgpuScale",
            (builder, b) =>
            {
                var kernel = builder.AddKernel(scaleMethod, descriptor);
                kernel.GridDimX = (uint)((N + 255) / 256);

                var inPort = builder.Input<float>("In", kernel.In(0));
                var outPort = builder.Output<float>("Out", kernel.Out(1));

                b.SetInputs(new() { inPort });
                b.SetOutputs(new() { outPort });
            });

        _engine.Context.SetExternalBuffer(block.Id, "In", bufIn.Pointer);
        _engine.Context.SetExternalBuffer(block.Id, "Out", bufOut.Pointer);

        _engine.Update();

        Assert.True(_engine.IsCompiled);

        var result = bufOut.Download();
        for (int i = 0; i < N; i++)
        {
            Assert.Equal(dataIn[i] * 2.0f, result[i], 0.001f);
        }
    }

    [Fact]
    public void MixedGraph_FilesystemAndIlgpu()
    {
        const int N = 256;

        // Block 1: ILGPU kernel (scale by 2)
        // Block 2: Filesystem PTX kernel (scalar_mul with scale=5)
        // Pipeline: input -> ILGPU -> connect -> ScalarMul -> output
        // Expected: input * 2 * 5 = input * 10

        using var bufIn = GpuBuffer<float>.Allocate(_device, N);
        using var bufOut = GpuBuffer<float>.Allocate(_device, N);

        var dataIn = Enumerable.Range(0, N).Select(i => 1.0f).ToArray();
        bufIn.Upload(dataIn);

        var scaleMethod = typeof(PatchableKernelTests).GetMethod(nameof(ScaleBy2Kernel),
            System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)!;

        var ilgpuDescriptor = MakeDescriptor("ScaleBy2Kernel",
            ("data", "float*", true, ParamDirection.In),
            ("output", "float*", true, ParamDirection.Out));

        // Block 1: ILGPU
        BlockPort? ilgpuOutPort = null;
        var ilgpuBlock = new BuiltTestBlock(_engine.Context, ScalarMulPath, "IlgpuBlock",
            (builder, b) =>
            {
                var kernel = builder.AddKernel(scaleMethod, ilgpuDescriptor);
                kernel.GridDimX = (uint)((N + 255) / 256);

                var inPort = builder.Input<float>("In", kernel.In(0));
                ilgpuOutPort = builder.Output<float>("Out", kernel.Out(1));

                b.SetInputs(new() { inPort });
                b.SetOutputs(new() { ilgpuOutPort });
            });

        // Block 2: Filesystem PTX (scalar_mul)
        BlockPort? mulInPort = null;
        var mulBlock = new BuiltTestBlock(_engine.Context, ScalarMulPath, "MulBlock",
            (builder, b) =>
            {
                var kernel = builder.AddKernel(ScalarMulPath);
                kernel.GridDimX = (uint)((N + 255) / 256);

                mulInPort = builder.Input<float>("A", kernel.In(0));
                var outPort = builder.Output<float>("C", kernel.Out(1));
                var scaleParam = builder.InputScalar<float>("scale", kernel.In(2), 5.0f);
                var countParam = builder.InputScalar<uint>("N", kernel.In(3), (uint)N);

                b.SetInputs(new() { mulInPort });
                b.SetOutputs(new() { outPort });
                b.SetParameters(new() { scaleParam, countParam });
            });

        // External buffers
        _engine.Context.SetExternalBuffer(ilgpuBlock.Id, "In", bufIn.Pointer);
        _engine.Context.SetExternalBuffer(mulBlock.Id, "C", bufOut.Pointer);

        // Connect: ILGPU.Out -> ScalarMul.A
        _engine.Context.Connect(ilgpuBlock.Id, "Out", mulBlock.Id, "A");

        _engine.Update();

        Assert.True(_engine.IsCompiled);

        // 1 * 2 * 5 = 10
        var result = bufOut.Download();
        for (int i = 0; i < N; i++)
        {
            Assert.Equal(10.0f, result[i], 0.001f);
        }
    }

    // --- NVRTC Tests (skip if DLL not available) ---

    [Fact(Skip = "NVRTC DLL (nvrtc64_130_0) requires CUDA Toolkit 13.0+ installed")]
    public void NvrtcKernel_CompilesAndExecutes()
    {
        const int N = 256;

        using var bufIn = GpuBuffer<float>.Allocate(_device, N);
        using var bufOut = GpuBuffer<float>.Allocate(_device, N);

        var dataIn = Enumerable.Range(0, N).Select(i => (float)i).ToArray();
        bufIn.Upload(dataIn);

        var descriptor = MakeDescriptor("nvrtc_scale",
            ("data", "float*", true, ParamDirection.In),
            ("output", "float*", true, ParamDirection.Out),
            ("scale", "float", false, ParamDirection.In),
            ("N", "uint", false, ParamDirection.In));

        var block = new BuiltTestBlock(_engine.Context, ScalarMulPath, "NvrtcScale",
            (builder, b) =>
            {
                var kernel = builder.AddKernelFromCuda(NvrtcScaleSource, "nvrtc_scale", descriptor);
                kernel.GridDimX = (uint)((N + 255) / 256);

                var inPort = builder.Input<float>("In", kernel.In(0));
                var outPort = builder.Output<float>("Out", kernel.Out(1));
                var scaleParam = builder.InputScalar<float>("scale", kernel.In(2), 3.0f);
                var countParam = builder.InputScalar<uint>("N", kernel.In(3), (uint)N);

                b.SetInputs(new() { inPort });
                b.SetOutputs(new() { outPort });
                b.SetParameters(new() { scaleParam, countParam });
            });

        _engine.Context.SetExternalBuffer(block.Id, "In", bufIn.Pointer);
        _engine.Context.SetExternalBuffer(block.Id, "Out", bufOut.Pointer);

        _engine.Update();

        Assert.True(_engine.IsCompiled);

        var result = bufOut.Download();
        for (int i = 0; i < N; i++)
        {
            Assert.Equal(dataIn[i] * 3.0f, result[i], 0.001f);
        }
    }

    [Fact(Skip = "NVRTC DLL (nvrtc64_130_0) requires CUDA Toolkit 13.0+ installed")]
    public void MixedGraph_NvrtcAndFilesystem()
    {
        const int N = 256;

        using var bufIn = GpuBuffer<float>.Allocate(_device, N);
        using var bufOut = GpuBuffer<float>.Allocate(_device, N);

        var dataIn = Enumerable.Range(0, N).Select(i => 2.0f).ToArray();
        bufIn.Upload(dataIn);

        var nvrtcDescriptor = MakeDescriptor("nvrtc_scale",
            ("data", "float*", true, ParamDirection.In),
            ("output", "float*", true, ParamDirection.Out),
            ("scale", "float", false, ParamDirection.In),
            ("N", "uint", false, ParamDirection.In));

        // Block 1: NVRTC kernel (scale by 4)
        BlockPort? nvrtcOutPort = null;
        var nvrtcBlock = new BuiltTestBlock(_engine.Context, ScalarMulPath, "NvrtcBlock",
            (builder, b) =>
            {
                var kernel = builder.AddKernelFromCuda(NvrtcScaleSource, "nvrtc_scale", nvrtcDescriptor);
                kernel.GridDimX = (uint)((N + 255) / 256);

                var inPort = builder.Input<float>("In", kernel.In(0));
                nvrtcOutPort = builder.Output<float>("Out", kernel.Out(1));
                var scaleParam = builder.InputScalar<float>("scale", kernel.In(2), 4.0f);
                var countParam = builder.InputScalar<uint>("N", kernel.In(3), (uint)N);

                b.SetInputs(new() { inPort });
                b.SetOutputs(new() { nvrtcOutPort });
                b.SetParameters(new() { scaleParam, countParam });
            });

        // Block 2: Filesystem PTX (scalar_mul, scale=3)
        BlockPort? mulInPort = null;
        var mulBlock = new BuiltTestBlock(_engine.Context, ScalarMulPath, "MulBlock",
            (builder, b) =>
            {
                var kernel = builder.AddKernel(ScalarMulPath);
                kernel.GridDimX = (uint)((N + 255) / 256);

                mulInPort = builder.Input<float>("A", kernel.In(0));
                var outPort = builder.Output<float>("C", kernel.Out(1));
                var scaleParam = builder.InputScalar<float>("scale", kernel.In(2), 3.0f);
                var countParam = builder.InputScalar<uint>("N", kernel.In(3), (uint)N);

                b.SetInputs(new() { mulInPort });
                b.SetOutputs(new() { outPort });
                b.SetParameters(new() { scaleParam, countParam });
            });

        _engine.Context.SetExternalBuffer(nvrtcBlock.Id, "In", bufIn.Pointer);
        _engine.Context.SetExternalBuffer(mulBlock.Id, "C", bufOut.Pointer);
        _engine.Context.Connect(nvrtcBlock.Id, "Out", mulBlock.Id, "A");

        _engine.Update();
        Assert.True(_engine.IsCompiled);

        // 2 * 4 * 3 = 24
        var result = bufOut.Download();
        for (int i = 0; i < N; i++)
        {
            Assert.Equal(24.0f, result[i], 0.001f);
        }
    }

    // --- BlockBuilder source type tests ---

    [Fact]
    public void BlockBuilder_AddKernel_IlgpuSource_StoresCorrectSource()
    {
        var scaleMethod = typeof(PatchableKernelTests).GetMethod(nameof(ScaleBy2Kernel),
            System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)!;

        var descriptor = MakeDescriptor("ScaleBy2Kernel",
            ("data", "float*", true, ParamDirection.In),
            ("output", "float*", true, ParamDirection.Out));

        var block = new TestBlock("TestIlgpu");
        _engine.Context.RegisterBlock(block);

        var builder = new BlockBuilder(_engine.Context, block);
        var handle = builder.AddKernel(scaleMethod, descriptor);

        Assert.IsType<KernelSource.IlgpuMethod>(handle.Source);
        var ilgpuSource = (KernelSource.IlgpuMethod)handle.Source;
        Assert.Same(scaleMethod, ilgpuSource.KernelMethod);
    }

    [Fact(Skip = "NVRTC DLL (nvrtc64_130_0) requires CUDA Toolkit 13.0+ installed")]
    public void BlockBuilder_AddKernelFromCuda_StoresCorrectSource()
    {
        var descriptor = MakeDescriptor("nvrtc_scale",
            ("data", "float*", true, ParamDirection.In),
            ("output", "float*", true, ParamDirection.Out),
            ("scale", "float", false, ParamDirection.In),
            ("N", "uint", false, ParamDirection.In));

        var block = new TestBlock("TestNvrtc");
        _engine.Context.RegisterBlock(block);

        var builder = new BlockBuilder(_engine.Context, block);
        var handle = builder.AddKernelFromCuda(NvrtcScaleSource, "nvrtc_scale", descriptor);

        Assert.IsType<KernelSource.NvrtcSource>(handle.Source);
        var nvrtcSource = (KernelSource.NvrtcSource)handle.Source;
        Assert.Equal("nvrtc_scale", nvrtcSource.EntryPoint);
        Assert.Equal(NvrtcScaleSource, nvrtcSource.CudaSource);
    }

    [Fact]
    public void BlockBuilder_AddKernel_FilesystemPtx_StoresCorrectSource()
    {
        var block = new TestBlock("TestFs");
        _engine.Context.RegisterBlock(block);

        var builder = new BlockBuilder(_engine.Context, block);
        var handle = builder.AddKernel(ScalarMulPath);

        Assert.IsType<KernelSource.FilesystemPtx>(handle.Source);
        var fsSource = (KernelSource.FilesystemPtx)handle.Source;
        Assert.Equal(ScalarMulPath, fsSource.PtxPath);
    }

    public void Dispose()
    {
        _engine.Dispose();
    }
}
