using System;
using ManagedCuda.BasicTypes;
using VL.Cuda.Core.Blocks.Builder;
using VL.Cuda.Core.Graph;
using Xunit;

namespace VL.Cuda.Tests.Graph;

public class CapturedNodeTests
{
    [Fact]
    public void Descriptor_DefaultsToEmptyLists()
    {
        var desc = new CapturedNodeDescriptor("test");

        Assert.Equal("test", desc.DebugName);
        Assert.Empty(desc.Inputs);
        Assert.Empty(desc.Outputs);
        Assert.Empty(desc.Scalars);
        Assert.Equal(0, desc.TotalParamCount);
    }

    [Fact]
    public void Descriptor_WithInputsAndOutputs()
    {
        var desc = new CapturedNodeDescriptor("blas.Sgemm",
            inputs: new[] { CapturedParam.Pointer("A", "float*"), CapturedParam.Pointer("B", "float*") },
            outputs: new[] { CapturedParam.Pointer("C", "float*") });

        Assert.Equal(2, desc.Inputs.Count);
        Assert.Single(desc.Outputs);
        Assert.Empty(desc.Scalars);
        Assert.Equal(3, desc.TotalParamCount);
    }

    [Fact]
    public void Descriptor_WithScalars()
    {
        var desc = new CapturedNodeDescriptor("blas.Sscal",
            outputs: new[] { CapturedParam.Pointer("x", "float*") },
            scalars: new[] { CapturedParam.Scalar("alpha", "float") });

        Assert.Single(desc.Outputs);
        Assert.Single(desc.Scalars);
        Assert.Equal(2, desc.TotalParamCount);
    }

    [Fact]
    public void CapturedParam_Pointer_IsPointerTrue()
    {
        var p = CapturedParam.Pointer("data", "float*");

        Assert.Equal("data", p.Name);
        Assert.Equal("float*", p.Type);
        Assert.True(p.IsPointer);
    }

    [Fact]
    public void CapturedParam_Scalar_IsPointerFalse()
    {
        var p = CapturedParam.Scalar("alpha", "float");

        Assert.Equal("alpha", p.Name);
        Assert.Equal("float", p.Type);
        Assert.False(p.IsPointer);
    }

    [Fact]
    public void CapturedNode_CreatesUniqueId()
    {
        var desc = new CapturedNodeDescriptor("test",
            outputs: new[] { CapturedParam.Pointer("out", "float*") });
        Action<CUstream, CUdeviceptr[]> action = (s, b) => { };

        var node = new CapturedNode(desc, action);

        Assert.NotEqual(Guid.Empty, node.Id);
        Assert.Equal("test", node.DebugName);
        Assert.Same(desc, node.Descriptor);
    }

    [Fact]
    public void CapturedNode_BufferBindingsMatchDescriptorSize()
    {
        var desc = new CapturedNodeDescriptor("test",
            inputs: new[] { CapturedParam.Pointer("A", "float*") },
            outputs: new[] { CapturedParam.Pointer("C", "float*") });
        Action<CUstream, CUdeviceptr[]> action = (s, b) => { };

        var node = new CapturedNode(desc, action);

        Assert.Equal(2, node.BufferBindings.Length);
    }

    [Fact]
    public void CapturedNode_HasCapturedGraph_FalseBeforeCapture()
    {
        var desc = new CapturedNodeDescriptor("test");
        var node = new CapturedNode(desc, (s, b) => { });

        Assert.False(node.HasCapturedGraph);
    }

    [Fact]
    public void CapturedNode_Dispose_DoesNotThrow()
    {
        var desc = new CapturedNodeDescriptor("test");
        var node = new CapturedNode(desc, (s, b) => { });

        node.Dispose();
        node.Dispose(); // double dispose is safe
    }

    [Fact]
    public void CapturedHandle_CreatesUniqueId()
    {
        var desc = new CapturedNodeDescriptor("test",
            inputs: new[] { CapturedParam.Pointer("A", "float*") },
            outputs: new[] { CapturedParam.Pointer("C", "float*") });
        Action<CUstream, CUdeviceptr[]> action = (s, b) => { };

        var handle = new CapturedHandle(desc, action);

        Assert.NotEqual(Guid.Empty, handle.Id);
        Assert.Same(desc, handle.Descriptor);
    }

    [Fact]
    public void CapturedHandle_InOut_ReturnsFlatIndices()
    {
        // Descriptor: 2 inputs, 1 output, 1 scalar → flat layout [in0, in1, out0, scl0]
        var desc = new CapturedNodeDescriptor("test",
            inputs: new[] { CapturedParam.Pointer("A", "float*"), CapturedParam.Pointer("B", "float*") },
            outputs: new[] { CapturedParam.Pointer("C", "float*") },
            scalars: new[] { CapturedParam.Scalar("alpha", "float") });
        var handle = new CapturedHandle(desc, (s, b) => { });

        var in0 = handle.In(0);   // flat index 0
        var in1 = handle.In(1);   // flat index 1
        var out0 = handle.Out(0); // flat index 2 (inputs.Count + 0)
        var scl0 = handle.Scalar(0); // flat index 3 (inputs.Count + outputs.Count + 0)

        Assert.Equal(handle.Id, in0.CapturedHandleId);
        Assert.Equal(0, in0.ParamIndex);
        Assert.Equal(CapturedPinCategory.Input, in0.Category);

        Assert.Equal(1, in1.ParamIndex);
        Assert.Equal(CapturedPinCategory.Input, in1.Category);

        Assert.Equal(handle.Id, out0.CapturedHandleId);
        Assert.Equal(2, out0.ParamIndex); // flat index: 2 inputs + 0
        Assert.Equal(CapturedPinCategory.Output, out0.Category);

        Assert.Equal(handle.Id, scl0.CapturedHandleId);
        Assert.Equal(3, scl0.ParamIndex); // flat index: 2 inputs + 1 output + 0
        Assert.Equal(CapturedPinCategory.Scalar, scl0.Category);
    }

    [Fact]
    public void CapturedNode_NullDescriptor_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new CapturedNode(null!, (s, b) => { }));
    }

    [Fact]
    public void CapturedNode_NullAction_Throws()
    {
        var desc = new CapturedNodeDescriptor("test");
        Assert.Throws<ArgumentNullException>(() =>
            new CapturedNode(desc, null!));
    }

    [Fact]
    public void CapturedHandle_NullDescriptor_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new CapturedHandle(null!, (s, b) => { }));
    }

    [Fact]
    public void CapturedHandle_NullAction_Throws()
    {
        var desc = new CapturedNodeDescriptor("test");
        Assert.Throws<ArgumentNullException>(() =>
            new CapturedHandle(desc, null!));
    }
}
