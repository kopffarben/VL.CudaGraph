namespace VL.Cuda.Core.Context;

/// <summary>
/// Configuration options for CudaEngine and the underlying CudaContext.
/// </summary>
public sealed class CudaEngineOptions
{
    /// <summary>
    /// CUDA device ordinal (default: 0 = first GPU).
    /// </summary>
    public int DeviceId { get; init; }
}
