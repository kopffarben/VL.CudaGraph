using System;
using ManagedCuda.BasicTypes;
using VL.Cuda.Core.Graph;

namespace VL.Cuda.Core.Blocks.Builder;

/// <summary>
/// Represents a captured library operation added to a block via BlockBuilder.AddCaptured().
/// Provides In()/Out()/Scalar() to reference specific parameters for port binding.
/// Analogous to KernelHandle for kernel operations.
/// </summary>
public sealed class CapturedHandle
{
    public Guid Id { get; }
    public CapturedNodeDescriptor Descriptor { get; }
    public Action<CUstream, CUdeviceptr[]> CaptureAction { get; }

    public CapturedHandle(CapturedNodeDescriptor descriptor, Action<CUstream, CUdeviceptr[]> captureAction)
    {
        Id = Guid.NewGuid();
        Descriptor = descriptor ?? throw new ArgumentNullException(nameof(descriptor));
        CaptureAction = captureAction ?? throw new ArgumentNullException(nameof(captureAction));
    }

    /// <summary>
    /// Reference to an input parameter by index. Returns the flat buffer binding index.
    /// Buffer bindings layout: [inputs..., outputs..., scalars...].
    /// </summary>
    public CapturedPin In(int index) => new(Id, index, CapturedPinCategory.Input);

    /// <summary>
    /// Reference to an output parameter by index. Returns the flat buffer binding index.
    /// Buffer bindings layout: [inputs..., outputs..., scalars...].
    /// </summary>
    public CapturedPin Out(int index) => new(Id, Descriptor.Inputs.Count + index, CapturedPinCategory.Output);

    /// <summary>
    /// Reference to a scalar parameter by index. Returns the flat buffer binding index.
    /// Buffer bindings layout: [inputs..., outputs..., scalars...].
    /// </summary>
    public CapturedPin Scalar(int index) => new(Id, Descriptor.Inputs.Count + Descriptor.Outputs.Count + index, CapturedPinCategory.Scalar);
}
