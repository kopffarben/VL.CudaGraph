using System;
using System.IO;
using System.Linq;
using ManagedCuda.BasicTypes;
using VL.Cuda.Core.Blocks;
using VL.Cuda.Core.Blocks.Builder;
using VL.Cuda.Core.Buffers;
using VL.Cuda.Core.Context;
using VL.Cuda.Core.Device;
using VL.Cuda.Core.Engine;
using VL.Cuda.Tests.Helpers;
using Xunit;

namespace VL.Cuda.Tests.Engine;

public class CudaEngineTests : IDisposable
{
    private readonly DeviceContext _device = new(0);
    private readonly CudaEngine _engine;

    private static string VectorAddPath => Path.Combine(AppContext.BaseDirectory, "TestKernels", "vector_add.ptx");
    private static string ScalarMulPath => Path.Combine(AppContext.BaseDirectory, "TestKernels", "scalar_mul.ptx");

    public CudaEngineTests()
    {
        _engine = new CudaEngine(new CudaContext(_device));
    }

    [Fact]
    public void NoBlocks_UpdateDoesNotThrow()
    {
        // First update with no blocks should not crash
        _engine.Update();
        Assert.False(_engine.IsCompiled);
    }

    [Fact]
    public void SingleBlock_CompilesAndLaunches()
    {
        const int N = 256;

        // Create block with vector_add kernel
        using var bufA = GpuBuffer<float>.Allocate(_device, N);
        using var bufB = GpuBuffer<float>.Allocate(_device, N);
        using var bufC = GpuBuffer<float>.Allocate(_device, N);

        // Upload test data
        var dataA = Enumerable.Range(0, N).Select(i => (float)i).ToArray();
        var dataB = Enumerable.Range(0, N).Select(i => (float)(i * 2)).ToArray();
        bufA.Upload(dataA);
        bufB.Upload(dataB);

        // Build block
        BlockParameter<uint>? countParam = null;
        var block = new BuiltTestBlock(_engine.Context, VectorAddPath, "VectorAdd",
            (builder, b) =>
            {
                var kernel = builder.AddKernel(VectorAddPath);
                kernel.GridDimX = (uint)((N + 255) / 256);

                var inA = builder.Input<float>("A", kernel.In(0));
                var inB = builder.Input<float>("B", kernel.In(1));
                var outC = builder.Output<float>("C", kernel.Out(2));
                countParam = builder.InputScalar<uint>("N", kernel.In(3), (uint)N);

                b.SetInputs(new() { inA, inB });
                b.SetOutputs(new() { outC });
                b.SetParameters(new() { countParam });
            });

        // Bind external buffers
        _engine.Context.SetExternalBuffer(block.Id, "A", bufA.Pointer);
        _engine.Context.SetExternalBuffer(block.Id, "B", bufB.Pointer);
        _engine.Context.SetExternalBuffer(block.Id, "C", bufC.Pointer);

        // First update: Cold Rebuild + Launch
        _engine.Update();

        Assert.True(_engine.IsCompiled);

        // Verify result
        var result = bufC.Download();
        for (int i = 0; i < N; i++)
        {
            Assert.Equal(dataA[i] + dataB[i], result[i], 0.001f);
        }
    }

    [Fact]
    public void TwoBlocksConnected_CompilesAndLaunches()
    {
        const int N = 256;

        // Create external buffers
        using var bufA = GpuBuffer<float>.Allocate(_device, N);
        using var bufB = GpuBuffer<float>.Allocate(_device, N);
        using var bufFinal = GpuBuffer<float>.Allocate(_device, N);

        var dataA = Enumerable.Range(0, N).Select(i => 1.0f).ToArray();
        var dataB = Enumerable.Range(0, N).Select(i => 2.0f).ToArray();
        bufA.Upload(dataA);
        bufB.Upload(dataB);

        // Block 1: vector_add (A + B → Sum)
        BlockPort? addOutPort = null;
        var addBlock = new BuiltTestBlock(_engine.Context, VectorAddPath, "Add",
            (builder, b) =>
            {
                var kernel = builder.AddKernel(VectorAddPath);
                kernel.GridDimX = (uint)((N + 255) / 256);

                var inA = builder.Input<float>("A", kernel.In(0));
                var inB = builder.Input<float>("B", kernel.In(1));
                addOutPort = builder.Output<float>("Sum", kernel.Out(2));
                var countP = builder.InputScalar<uint>("N", kernel.In(3), (uint)N);

                b.SetInputs(new() { inA, inB });
                b.SetOutputs(new() { addOutPort });
                b.SetParameters(new() { countP });
            });

        // Block 2: scalar_mul (Sum * scale → Result)
        BlockParameter<float>? scaleParam = null;
        BlockPort? mulInPort = null;
        var mulBlock = new BuiltTestBlock(_engine.Context, ScalarMulPath, "Mul",
            (builder, b) =>
            {
                var kernel = builder.AddKernel(ScalarMulPath);
                kernel.GridDimX = (uint)((N + 255) / 256);

                mulInPort = builder.Input<float>("A", kernel.In(0));
                var outC = builder.Output<float>("C", kernel.Out(1));
                scaleParam = builder.InputScalar<float>("scale", kernel.In(2), 10.0f);
                var countP = builder.InputScalar<uint>("N", kernel.In(3), (uint)N);

                b.SetInputs(new() { mulInPort });
                b.SetOutputs(new() { outC });
                b.SetParameters(new() { scaleParam, countP });
            });

        // External buffers
        _engine.Context.SetExternalBuffer(addBlock.Id, "A", bufA.Pointer);
        _engine.Context.SetExternalBuffer(addBlock.Id, "B", bufB.Pointer);
        _engine.Context.SetExternalBuffer(mulBlock.Id, "C", bufFinal.Pointer);

        // Connect: add.Sum → mul.A
        _engine.Context.Connect(addBlock.Id, "Sum", mulBlock.Id, "A");

        // Launch
        _engine.Update();

        Assert.True(_engine.IsCompiled);

        // Verify: (1 + 2) * 10 = 30
        var result = bufFinal.Download();
        for (int i = 0; i < N; i++)
        {
            Assert.Equal(30.0f, result[i], 0.001f);
        }
    }

    [Fact]
    public void HotUpdate_ScalarChange_NoRebuild()
    {
        const int N = 256;

        using var bufA = GpuBuffer<float>.Allocate(_device, N);
        using var bufC = GpuBuffer<float>.Allocate(_device, N);

        var dataA = Enumerable.Range(0, N).Select(i => 1.0f).ToArray();
        bufA.Upload(dataA);

        BlockParameter<float>? scaleParam = null;
        var block = new BuiltTestBlock(_engine.Context, ScalarMulPath, "Mul",
            (builder, b) =>
            {
                var kernel = builder.AddKernel(ScalarMulPath);
                kernel.GridDimX = (uint)((N + 255) / 256);

                var inA = builder.Input<float>("A", kernel.In(0));
                var outC = builder.Output<float>("C", kernel.Out(1));
                scaleParam = builder.InputScalar<float>("scale", kernel.In(2), 5.0f);
                var countP = builder.InputScalar<uint>("N", kernel.In(3), (uint)N);

                b.SetInputs(new() { inA });
                b.SetOutputs(new() { outC });
                b.SetParameters(new() { scaleParam, countP });
            });

        _engine.Context.SetExternalBuffer(block.Id, "A", bufA.Pointer);
        _engine.Context.SetExternalBuffer(block.Id, "C", bufC.Pointer);

        // First update: Cold Rebuild
        _engine.Update();
        var firstRebuildTime = _engine.LastRebuildTime;

        // Verify: 1 * 5 = 5
        var result1 = bufC.Download();
        Assert.Equal(5.0f, result1[0], 0.001f);

        // Change scalar parameter (Hot Update, no rebuild)
        scaleParam!.TypedValue = 3.0f;

        Assert.True(_engine.Context.Dirty.AreParametersDirty);
        Assert.False(_engine.Context.Dirty.IsStructureDirty);

        // Second update: Hot Update + Launch
        _engine.Update();

        // Verify: 1 * 3 = 3
        var result2 = bufC.Download();
        Assert.Equal(3.0f, result2[0], 0.001f);
    }

    [Fact]
    public void ColdRebuild_WhenBlockAdded()
    {
        // First update with no blocks
        _engine.Update();
        Assert.False(_engine.IsCompiled);

        const int N = 256;
        using var bufA = GpuBuffer<float>.Allocate(_device, N);
        using var bufC = GpuBuffer<float>.Allocate(_device, N);
        bufA.Upload(Enumerable.Range(0, N).Select(i => 2.0f).ToArray());

        // Add a block
        var block = new BuiltTestBlock(_engine.Context, ScalarMulPath, "Mul",
            (builder, b) =>
            {
                var kernel = builder.AddKernel(ScalarMulPath);
                kernel.GridDimX = (uint)((N + 255) / 256);

                var inA = builder.Input<float>("A", kernel.In(0));
                var outC = builder.Output<float>("C", kernel.Out(1));
                var scaleP = builder.InputScalar<float>("scale", kernel.In(2), 4.0f);
                var countP = builder.InputScalar<uint>("N", kernel.In(3), (uint)N);

                b.SetInputs(new() { inA });
                b.SetOutputs(new() { outC });
                b.SetParameters(new() { scaleP, countP });
            });

        _engine.Context.SetExternalBuffer(block.Id, "A", bufA.Pointer);
        _engine.Context.SetExternalBuffer(block.Id, "C", bufC.Pointer);

        Assert.True(_engine.Context.Dirty.IsStructureDirty);

        // Second update: Cold Rebuild
        _engine.Update();

        Assert.True(_engine.IsCompiled);

        var result = bufC.Download();
        Assert.Equal(8.0f, result[0], 0.001f);
    }

    [Fact]
    public void ColdRebuild_WhenBlockRemoved()
    {
        const int N = 256;
        using var bufA = GpuBuffer<float>.Allocate(_device, N);
        using var bufC = GpuBuffer<float>.Allocate(_device, N);
        bufA.Upload(Enumerable.Range(0, N).Select(i => 1.0f).ToArray());

        var block = new BuiltTestBlock(_engine.Context, ScalarMulPath, "Mul",
            (builder, b) =>
            {
                var kernel = builder.AddKernel(ScalarMulPath);
                kernel.GridDimX = 1;

                var inA = builder.Input<float>("A", kernel.In(0));
                var outC = builder.Output<float>("C", kernel.Out(1));
                var scaleP = builder.InputScalar<float>("scale", kernel.In(2), 1.0f);
                var countP = builder.InputScalar<uint>("N", kernel.In(3), (uint)N);

                b.SetInputs(new() { inA });
                b.SetOutputs(new() { outC });
                b.SetParameters(new() { scaleP, countP });
            });

        _engine.Context.SetExternalBuffer(block.Id, "A", bufA.Pointer);
        _engine.Context.SetExternalBuffer(block.Id, "C", bufC.Pointer);

        // First update: compiles and launches
        _engine.Update();
        Assert.True(_engine.IsCompiled);

        // Remove block → structure dirty
        block.Dispose();

        Assert.True(_engine.Context.Dirty.IsStructureDirty);

        // Second update: Cold Rebuild (no blocks → not compiled)
        _engine.Update();
        Assert.False(_engine.IsCompiled);
    }

    [Fact]
    public void DebugInfo_DistributedAfterLaunch()
    {
        const int N = 256;
        using var bufA = GpuBuffer<float>.Allocate(_device, N);
        using var bufC = GpuBuffer<float>.Allocate(_device, N);
        bufA.Upload(new float[N]);

        var block = new BuiltTestBlock(_engine.Context, ScalarMulPath, "Mul",
            (builder, b) =>
            {
                var kernel = builder.AddKernel(ScalarMulPath);
                kernel.GridDimX = 1;

                var inA = builder.Input<float>("A", kernel.In(0));
                var outC = builder.Output<float>("C", kernel.Out(1));
                var scaleP = builder.InputScalar<float>("scale", kernel.In(2), 1.0f);
                var countP = builder.InputScalar<uint>("N", kernel.In(3), (uint)N);

                b.SetInputs(new() { inA });
                b.SetOutputs(new() { outC });
                b.SetParameters(new() { scaleP, countP });
            });

        _engine.Context.SetExternalBuffer(block.Id, "A", bufA.Pointer);
        _engine.Context.SetExternalBuffer(block.Id, "C", bufC.Pointer);

        _engine.Update();

        Assert.NotNull(block.DebugInfo);
        Assert.Equal(BlockState.OK, block.DebugInfo.State);
    }

    [Fact]
    public void Dispose_CleansUp()
    {
        _engine.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _engine.Update());
    }

    public void Dispose()
    {
        _engine.Dispose();
    }
}
