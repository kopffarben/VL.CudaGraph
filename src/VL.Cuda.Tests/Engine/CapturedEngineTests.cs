using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using VL.Cuda.Core.Blocks;
using VL.Cuda.Core.Blocks.Builder;
using VL.Cuda.Core.Buffers;
using VL.Cuda.Core.Device;
using VL.Cuda.Core.Engine;
using VL.Cuda.Core.Graph;
using CudaContext = VL.Cuda.Core.Context.CudaContext;
using VL.Cuda.Tests.Helpers;
using Xunit;

namespace VL.Cuda.Tests.Engine;

public class CapturedEngineTests : IDisposable
{
    private readonly DeviceContext _device = new(0);
    private readonly CudaEngine _engine;

    private static string VectorAddPath => Path.Combine(AppContext.BaseDirectory, "TestKernels", "vector_add.ptx");

    public CapturedEngineTests()
    {
        _engine = new CudaEngine(new CudaContext(_device));
    }

    /// <summary>
    /// Create a block with a single captured operation that does a memset on its output buffer.
    /// This is the simplest possible captured node: fills output with a known value.
    /// </summary>
    private BuiltTestBlock CreateMemsetCapturedBlock(uint fillValue, int count)
    {
        return new BuiltTestBlock(_engine.Context, VectorAddPath, "CapturedMemset",
            (builder, b) =>
            {
                var desc = new CapturedNodeDescriptor("test.Memset",
                    outputs: new[] { CapturedParam.Pointer("Output", "uint*") });

                var handle = builder.AddCaptured((stream, buffers) =>
                {
                    // During capture, do an async memset on the output buffer
                    DriverAPINativeMethods.MemsetAsync.cuMemsetD32Async(
                        buffers[0], fillValue, (SizeT)count, stream);
                }, desc);

                var outPort = builder.Output<uint>("Output", handle.Out(0));
                b.SetOutputs(new List<IBlockPort> { outPort });
            });
    }

    [Fact]
    public void CapturedBlock_CompilesAndLaunches()
    {
        const int N = 64;
        const uint fillValue = 42;

        using var buf = GpuBuffer<uint>.Allocate(_device, N);
        buf.Upload(new uint[N]); // zero-fill

        var block = CreateMemsetCapturedBlock(fillValue, N);

        _engine.Context.SetExternalBuffer(block.Id, "Output", buf.Pointer);

        _engine.Update();

        Assert.True(_engine.IsCompiled);

        var result = buf.Download();
        for (int i = 0; i < N; i++)
        {
            Assert.Equal(fillValue, result[i]);
        }
    }

    [Fact]
    public void CapturedBlock_DebugInfoDistributed()
    {
        const int N = 16;
        using var buf = GpuBuffer<uint>.Allocate(_device, N);
        buf.Upload(new uint[N]);

        var block = CreateMemsetCapturedBlock(1u, N);
        _engine.Context.SetExternalBuffer(block.Id, "Output", buf.Pointer);

        _engine.Update();

        Assert.NotNull(block.DebugInfo);
        Assert.Equal(BlockState.OK, block.DebugInfo.State);
    }

    [Fact]
    public void CapturedBlock_ColdRebuild_WhenRemoved()
    {
        const int N = 16;
        using var buf = GpuBuffer<uint>.Allocate(_device, N);
        buf.Upload(new uint[N]);

        var block = CreateMemsetCapturedBlock(1u, N);
        _engine.Context.SetExternalBuffer(block.Id, "Output", buf.Pointer);

        _engine.Update();
        Assert.True(_engine.IsCompiled);

        // Remove block → triggers structure dirty
        block.Dispose();

        Assert.True(_engine.Context.Dirty.IsStructureDirty);

        _engine.Update();
        Assert.False(_engine.IsCompiled); // No blocks left
    }

    [Fact]
    public void MixedGraph_KernelAndCaptured_CompilesAndLaunches()
    {
        // This test creates two blocks:
        // 1. A kernel block (vector_add) that adds two input buffers
        // 2. A captured block that memsets its output
        // They are independent (no connections) but execute in the same graph.
        const int N = 256;
        const uint fillValue = 77;

        // Kernel block: vector_add
        using var bufA = GpuBuffer<float>.Allocate(_device, N);
        using var bufB = GpuBuffer<float>.Allocate(_device, N);
        using var bufC = GpuBuffer<float>.Allocate(_device, N);
        bufA.Upload(Enumerable.Range(0, N).Select(i => 1.0f).ToArray());
        bufB.Upload(Enumerable.Range(0, N).Select(i => 2.0f).ToArray());

        var kernelBlock = new BuiltTestBlock(_engine.Context, VectorAddPath, "VectorAdd",
            (builder, b) =>
            {
                var kernel = builder.AddKernel(VectorAddPath);
                kernel.GridDimX = (uint)((N + 255) / 256);

                var inA = builder.Input<float>("A", kernel.In(0));
                var inB = builder.Input<float>("B", kernel.In(1));
                var outC = builder.Output<float>("C", kernel.Out(2));
                var countP = builder.InputScalar<uint>("N", kernel.In(3), (uint)N);

                b.SetInputs(new List<IBlockPort> { inA, inB });
                b.SetOutputs(new List<IBlockPort> { outC });
                b.SetParameters(new List<IBlockParameter> { countP });
            });

        _engine.Context.SetExternalBuffer(kernelBlock.Id, "A", bufA.Pointer);
        _engine.Context.SetExternalBuffer(kernelBlock.Id, "B", bufB.Pointer);
        _engine.Context.SetExternalBuffer(kernelBlock.Id, "C", bufC.Pointer);

        // Captured block: memset
        using var bufOut = GpuBuffer<uint>.Allocate(_device, N);
        bufOut.Upload(new uint[N]);

        var capturedBlock = CreateMemsetCapturedBlock(fillValue, N);
        _engine.Context.SetExternalBuffer(capturedBlock.Id, "Output", bufOut.Pointer);

        // Update — compiles both blocks into one graph
        _engine.Update();

        Assert.True(_engine.IsCompiled);

        // Verify kernel block result: 1 + 2 = 3
        var resultC = bufC.Download();
        Assert.Equal(3.0f, resultC[0], 0.001f);

        // Verify captured block result: filled with 77
        var resultOut = bufOut.Download();
        Assert.Equal(fillValue, resultOut[0]);
    }

    [Fact]
    public void CapturedBlock_MultipleOutputs()
    {
        // A captured node with two output buffers
        const int N = 32;
        const uint fill1 = 11;
        const uint fill2 = 22;

        using var buf1 = GpuBuffer<uint>.Allocate(_device, N);
        using var buf2 = GpuBuffer<uint>.Allocate(_device, N);
        buf1.Upload(new uint[N]);
        buf2.Upload(new uint[N]);

        var block = new BuiltTestBlock(_engine.Context, VectorAddPath, "DualOutput",
            (builder, b) =>
            {
                var desc = new CapturedNodeDescriptor("test.DualMemset",
                    outputs: new[]
                    {
                        CapturedParam.Pointer("Out1", "uint*"),
                        CapturedParam.Pointer("Out2", "uint*"),
                    });

                var handle = builder.AddCaptured((stream, buffers) =>
                {
                    DriverAPINativeMethods.MemsetAsync.cuMemsetD32Async(
                        buffers[0], fill1, (SizeT)N, stream);
                    DriverAPINativeMethods.MemsetAsync.cuMemsetD32Async(
                        buffers[1], fill2, (SizeT)N, stream);
                }, desc);

                var out1 = builder.Output<uint>("Out1", handle.Out(0));
                var out2 = builder.Output<uint>("Out2", handle.Out(1));
                b.SetOutputs(new List<IBlockPort> { out1, out2 });
            });

        _engine.Context.SetExternalBuffer(block.Id, "Out1", buf1.Pointer);
        _engine.Context.SetExternalBuffer(block.Id, "Out2", buf2.Pointer);

        _engine.Update();
        Assert.True(_engine.IsCompiled);

        var result1 = buf1.Download();
        var result2 = buf2.Download();
        Assert.Equal(fill1, result1[0]);
        Assert.Equal(fill2, result2[0]);
    }

    [Fact]
    public void CapturedBlock_InputAndOutput()
    {
        // A captured node that reads an input and writes to an output
        // Uses cuMemcpyDtoDAsync to copy input → output during capture
        const int N = 16;

        using var bufIn = GpuBuffer<uint>.Allocate(_device, N);
        using var bufOut = GpuBuffer<uint>.Allocate(_device, N);
        var inputData = Enumerable.Range(0, N).Select(i => (uint)(i * 3)).ToArray();
        bufIn.Upload(inputData);
        bufOut.Upload(new uint[N]);

        var block = new BuiltTestBlock(_engine.Context, VectorAddPath, "Copy",
            (builder, b) =>
            {
                var desc = new CapturedNodeDescriptor("test.Copy",
                    inputs: new[] { CapturedParam.Pointer("Input", "uint*") },
                    outputs: new[] { CapturedParam.Pointer("Output", "uint*") });

                var handle = builder.AddCaptured((stream, buffers) =>
                {
                    // Copy input to output (descriptor order: inputs first, then outputs)
                    DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyDtoDAsync_v2(
                        buffers[1], buffers[0], (SizeT)(N * sizeof(uint)), stream);
                }, desc);

                var inPort = builder.Input<uint>("Input", handle.In(0));
                var outPort = builder.Output<uint>("Output", handle.Out(0));
                b.SetInputs(new List<IBlockPort> { inPort });
                b.SetOutputs(new List<IBlockPort> { outPort });
            });

        _engine.Context.SetExternalBuffer(block.Id, "Input", bufIn.Pointer);
        _engine.Context.SetExternalBuffer(block.Id, "Output", bufOut.Pointer);

        _engine.Update();
        Assert.True(_engine.IsCompiled);

        var result = bufOut.Download();
        for (int i = 0; i < N; i++)
        {
            Assert.Equal(inputData[i], result[i]);
        }
    }

    [Fact]
    public void RecaptureDirty_TriggersRecapture()
    {
        // This test verifies that marking a captured node dirty triggers Recapture path
        const int N = 16;

        using var buf = GpuBuffer<uint>.Allocate(_device, N);
        buf.Upload(new uint[N]);

        Guid handleId = Guid.Empty;
        var block = new BuiltTestBlock(_engine.Context, VectorAddPath, "RecaptureTest",
            (builder, b) =>
            {
                var desc = new CapturedNodeDescriptor("test.Memset",
                    outputs: new[] { CapturedParam.Pointer("Output", "uint*") });

                var handle = builder.AddCaptured((stream, buffers) =>
                {
                    DriverAPINativeMethods.MemsetAsync.cuMemsetD32Async(
                        buffers[0], 100u, (SizeT)N, stream);
                }, desc);

                handleId = handle.Id;
                var outPort = builder.Output<uint>("Output", handle.Out(0));
                b.SetOutputs(new List<IBlockPort> { outPort });
            });

        _engine.Context.SetExternalBuffer(block.Id, "Output", buf.Pointer);

        // First update: Cold Rebuild
        _engine.Update();
        Assert.True(_engine.IsCompiled);

        var result1 = buf.Download();
        Assert.Equal(100u, result1[0]);

        // Mark captured node dirty for Recapture
        _engine.Context.OnCapturedNodeChanged(block.Id, handleId);

        Assert.True(_engine.Context.Dirty.AreCapturedNodesDirty);
        Assert.False(_engine.Context.Dirty.IsStructureDirty);
        Assert.False(_engine.Context.Dirty.AreParametersDirty);

        // Second update: Recapture path (not Cold Rebuild)
        _engine.Update();

        // Result should still be 100 (same operation, recaptured)
        var result2 = buf.Download();
        Assert.Equal(100u, result2[0]);
    }

    [Fact]
    public void Dispose_CleansCapturedNodes()
    {
        const int N = 16;
        var buf = GpuBuffer<uint>.Allocate(_device, N);
        buf.Upload(new uint[N]);

        var block = CreateMemsetCapturedBlock(1u, N);
        _engine.Context.SetExternalBuffer(block.Id, "Output", buf.Pointer);

        _engine.Update();
        Assert.True(_engine.IsCompiled);

        // Dispose buffer before engine (engine dispose also disposes DeviceContext)
        buf.Dispose();
        _engine.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _engine.Update());
    }

    public void Dispose()
    {
        _engine.Dispose();
    }
}
