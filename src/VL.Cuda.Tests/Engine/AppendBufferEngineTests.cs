using System;
using System.IO;
using System.Linq;
using VL.Cuda.Core.Blocks;
using VL.Cuda.Core.Blocks.Builder;
using VL.Cuda.Core.Buffers;
using VL.Cuda.Core.Context;
using VL.Cuda.Core.Device;
using VL.Cuda.Core.Engine;
using VL.Cuda.Tests.Helpers;
using Xunit;

namespace VL.Cuda.Tests.Engine;

public class AppendBufferEngineTests : IDisposable
{
    private readonly DeviceContext _device = new(0);
    private readonly CudaEngine _engine;

    private static string AppendTestPath => Path.Combine(AppContext.BaseDirectory, "TestKernels", "append_test.ptx");

    public AppendBufferEngineTests()
    {
        _engine = new CudaEngine(new CudaContext(_device));
    }

    [Fact]
    public void Block_WithAppendOutput_CompilesAndLaunches()
    {
        const int N = 256;
        const int MaxCapacity = 256;

        using var bufInput = GpuBuffer<float>.Allocate(_device, N);

        // Upload test data: values 0..255
        var inputData = Enumerable.Range(0, N).Select(i => (float)i).ToArray();
        bufInput.Upload(inputData);

        // Build block with append output
        BlockParameter<float>? thresholdParam = null;
        BlockParameter<uint>? countParam = null;
        var block = new BuiltTestBlock(_engine.Context, AppendTestPath, "AppendFilter",
            (builder, b) =>
            {
                var kernel = builder.AddKernel(AppendTestPath);
                kernel.GridDimX = (uint)((N + 255) / 256);

                var inBuf = builder.Input<float>("input", kernel.In(0));

                // AppendOutput: data port (index 1) + counter port (index 2)
                var appendOut = builder.AppendOutput<float>("output", kernel.Out(1), kernel.Out(2), MaxCapacity);

                thresholdParam = builder.InputScalar<float>("threshold", kernel.In(3), 128.0f);
                countParam = builder.InputScalar<uint>("N", kernel.In(4), (uint)N);

                b.SetInputs(new() { inBuf });
                b.SetOutputs(new() { appendOut.DataPort });
                b.SetParameters(new() { thresholdParam, countParam });
            });

        _engine.Context.SetExternalBuffer(block.Id, "input", bufInput.Pointer);

        // Launch
        _engine.Update();

        Assert.True(_engine.IsCompiled);

        // Check debug info has AppendCounts
        Assert.NotNull(block.DebugInfo);
        Assert.Equal(BlockState.OK, block.DebugInfo.State);
        Assert.NotNull(block.DebugInfo.AppendCounts);
        Assert.True(block.DebugInfo.AppendCounts.ContainsKey("output"));
    }

    [Fact]
    public void CounterAutoRead_AfterLaunch()
    {
        const int N = 256;
        const int MaxCapacity = 256;

        using var bufInput = GpuBuffer<float>.Allocate(_device, N);

        // Input: 0..255. Threshold: 200. Expected: values 201..255 = 55 items
        var inputData = Enumerable.Range(0, N).Select(i => (float)i).ToArray();
        bufInput.Upload(inputData);

        var block = new BuiltTestBlock(_engine.Context, AppendTestPath, "AppendFilter",
            (builder, b) =>
            {
                var kernel = builder.AddKernel(AppendTestPath);
                kernel.GridDimX = (uint)((N + 255) / 256);

                var inBuf = builder.Input<float>("input", kernel.In(0));
                var appendOut = builder.AppendOutput<float>("output", kernel.Out(1), kernel.Out(2), MaxCapacity);
                var threshP = builder.InputScalar<float>("threshold", kernel.In(3), 200.0f);
                var countP = builder.InputScalar<uint>("N", kernel.In(4), (uint)N);

                b.SetInputs(new() { inBuf });
                b.SetOutputs(new() { appendOut.DataPort });
                b.SetParameters(new() { threshP, countP });
            });

        _engine.Context.SetExternalBuffer(block.Id, "input", bufInput.Pointer);

        _engine.Update();

        // Values > 200: 201, 202, ..., 255 = 55 items
        var count = block.DebugInfo.AppendCounts!["output"];
        Assert.Equal(55, count);
    }

    [Fact]
    public void CounterReset_OnSecondLaunch()
    {
        const int N = 256;
        const int MaxCapacity = 256;

        using var bufInput = GpuBuffer<float>.Allocate(_device, N);

        var inputData = Enumerable.Range(0, N).Select(i => (float)i).ToArray();
        bufInput.Upload(inputData);

        BlockParameter<float>? thresholdParam = null;
        var block = new BuiltTestBlock(_engine.Context, AppendTestPath, "AppendFilter",
            (builder, b) =>
            {
                var kernel = builder.AddKernel(AppendTestPath);
                kernel.GridDimX = (uint)((N + 255) / 256);

                var inBuf = builder.Input<float>("input", kernel.In(0));
                var appendOut = builder.AppendOutput<float>("output", kernel.Out(1), kernel.Out(2), MaxCapacity);
                thresholdParam = builder.InputScalar<float>("threshold", kernel.In(3), 200.0f);
                var countP = builder.InputScalar<uint>("N", kernel.In(4), (uint)N);

                b.SetInputs(new() { inBuf });
                b.SetOutputs(new() { appendOut.DataPort });
                b.SetParameters(new() { thresholdParam, countP });
            });

        _engine.Context.SetExternalBuffer(block.Id, "input", bufInput.Pointer);

        // First launch: 55 items
        _engine.Update();
        Assert.Equal(55, block.DebugInfo.AppendCounts!["output"]);

        // Second launch (same data): counter should have been reset by memset node
        // Changing threshold triggers parameter dirty, but we need a second frame
        thresholdParam!.TypedValue = 250.0f;
        _engine.Update();

        // Values > 250: 251, 252, 253, 254, 255 = 5 items
        Assert.Equal(5, block.DebugInfo.AppendCounts!["output"]);
    }

    [Fact]
    public void AllFiltered_ZeroCount()
    {
        const int N = 256;
        const int MaxCapacity = 256;

        using var bufInput = GpuBuffer<float>.Allocate(_device, N);

        // All values 0..255, threshold 999 → nothing passes
        var inputData = Enumerable.Range(0, N).Select(i => (float)i).ToArray();
        bufInput.Upload(inputData);

        var block = new BuiltTestBlock(_engine.Context, AppendTestPath, "AppendFilter",
            (builder, b) =>
            {
                var kernel = builder.AddKernel(AppendTestPath);
                kernel.GridDimX = (uint)((N + 255) / 256);

                var inBuf = builder.Input<float>("input", kernel.In(0));
                var appendOut = builder.AppendOutput<float>("output", kernel.Out(1), kernel.Out(2), MaxCapacity);
                var threshP = builder.InputScalar<float>("threshold", kernel.In(3), 999.0f);
                var countP = builder.InputScalar<uint>("N", kernel.In(4), (uint)N);

                b.SetInputs(new() { inBuf });
                b.SetOutputs(new() { appendOut.DataPort });
                b.SetParameters(new() { threshP, countP });
            });

        _engine.Context.SetExternalBuffer(block.Id, "input", bufInput.Pointer);

        _engine.Update();

        Assert.Equal(0, block.DebugInfo.AppendCounts!["output"]);
    }

    [Fact]
    public void AppendBufferInfo_CountPortName()
    {
        var info = new AppendBufferInfo(
            Guid.NewGuid(), "Particles",
            Guid.NewGuid(), 1,
            Guid.NewGuid(), 2,
            1024, 4);

        Assert.Equal("Particles Count", info.CountPortName);
    }

    [Fact]
    public void AppendBufferInfo_StructuralEquals()
    {
        var id = Guid.NewGuid();
        var h1 = Guid.NewGuid();
        var h2 = Guid.NewGuid();

        var a = new AppendBufferInfo(id, "Out", h1, 1, h2, 2, 1024, 4);
        var b = new AppendBufferInfo(id, "Out", Guid.NewGuid(), 1, Guid.NewGuid(), 2, 1024, 4);

        // StructuralEquals ignores HandleIds (they change per construction)
        Assert.True(a.StructuralEquals(b));

        // Different capacity → not equal
        var c = new AppendBufferInfo(id, "Out", h1, 1, h2, 2, 2048, 4);
        Assert.False(a.StructuralEquals(c));
    }

    public void Dispose()
    {
        _engine.Dispose();
    }
}
