using System;
using System.Runtime.InteropServices;
using ManagedCuda.BasicTypes;
using VL.Cuda.Core.PTX;

namespace VL.Cuda.Core.Graph;

/// <summary>
/// Describes a kernel in the CUDA graph. Stores the loaded kernel handle,
/// grid configuration, and parameter values in pinned native memory for
/// the CUDA Graph API.
/// </summary>
public sealed class KernelNode : IGraphNode, IDisposable
{
    private readonly IntPtr _paramBlock;   // array of IntPtr (one per parameter)
    private readonly IntPtr[] _paramSlots; // native alloc per parameter (value storage)
    private readonly int _paramCount;
    private bool _disposed;

    public Guid Id { get; }
    public string DebugName { get; set; }
    public LoadedKernel LoadedKernel { get; }

    // Grid configuration
    public uint BlockDimX { get; set; }
    public uint BlockDimY { get; set; }
    public uint BlockDimZ { get; set; }
    public uint GridDimX { get; set; }
    public uint GridDimY { get; set; }
    public uint GridDimZ { get; set; }
    public uint SharedMemoryBytes { get; set; }

    public KernelNode(LoadedKernel loadedKernel, string? debugName = null)
    {
        Id = Guid.NewGuid();
        DebugName = debugName ?? loadedKernel.Descriptor.EntryPoint;
        LoadedKernel = loadedKernel;

        var desc = loadedKernel.Descriptor;
        _paramCount = desc.Parameters.Count;

        // Default grid config from descriptor
        BlockDimX = desc.BlockSize > 0 ? (uint)desc.BlockSize : 256;
        BlockDimY = 1;
        BlockDimZ = 1;
        GridDimX = 1;
        GridDimY = 1;
        GridDimZ = 1;
        SharedMemoryBytes = desc.SharedMemoryBytes > 0 ? (uint)desc.SharedMemoryBytes : 0;

        // Allocate native param block: an array of IntPtr, one per parameter
        _paramBlock = Marshal.AllocHGlobal(_paramCount * IntPtr.Size);
        _paramSlots = new IntPtr[_paramCount];

        for (int i = 0; i < _paramCount; i++)
        {
            // Each slot needs enough space for the parameter.
            // Use explicit SizeBytes if set (e.g. 16 for ILGPU ArrayView structs),
            // otherwise: pointers = IntPtr.Size, scalars = type-based.
            var param = desc.Parameters[i];
            int slotSize = param.SizeBytes > 0
                ? param.SizeBytes
                : (param.IsPointer ? IntPtr.Size : GetScalarSize(param.Type));
            _paramSlots[i] = Marshal.AllocHGlobal(slotSize);

            // Zero-initialize
            for (int b = 0; b < slotSize; b++)
                Marshal.WriteByte(_paramSlots[i], b, 0);

            // Write slot address into the param block array
            Marshal.WriteIntPtr(_paramBlock, i * IntPtr.Size, _paramSlots[i]);
        }
    }

    /// <summary>
    /// Set a pointer parameter (buffer device pointer).
    /// </summary>
    public void SetPointer(int index, CUdeviceptr ptr)
    {
        ValidateIndex(index);
        // CUdeviceptr.Pointer is a SizeT which wraps a ulong
        Marshal.WriteIntPtr(_paramSlots[index], new IntPtr((long)(ulong)ptr.Pointer));
    }

    /// <summary>
    /// Set an ILGPU ArrayView struct parameter: writes the device pointer at offset 0
    /// and the length at offset 8 in the 16-byte struct slot.
    /// </summary>
    public unsafe void SetArrayView(int index, CUdeviceptr ptr, long length)
    {
        ValidateIndex(index);
        var slot = (byte*)_paramSlots[index];
        *(long*)slot = (long)(ulong)ptr.Pointer;
        *(long*)(slot + 8) = length;
    }

    /// <summary>
    /// Set a scalar parameter value.
    /// </summary>
    public unsafe void SetScalar<T>(int index, T value) where T : unmanaged
    {
        ValidateIndex(index);
        *(T*)_paramSlots[index] = value;
    }

    /// <summary>
    /// Get the IntPtr to the parameter array for CudaKernelNodeParams.kernelParams.
    /// </summary>
    public IntPtr GetParamsPtr() => _paramBlock;

    /// <summary>
    /// Build the CudaKernelNodeParams struct for this node.
    /// </summary>
    internal CudaKernelNodeParams BuildNodeParams()
    {
        return new CudaKernelNodeParams
        {
            func = LoadedKernel.Kernel.CUFunction,
            gridDimX = GridDimX,
            gridDimY = GridDimY,
            gridDimZ = GridDimZ,
            blockDimX = BlockDimX,
            blockDimY = BlockDimY,
            blockDimZ = BlockDimZ,
            sharedMemBytes = SharedMemoryBytes,
            kernelParams = _paramBlock,
            extra = IntPtr.Zero,
        };
    }

    public int ParameterCount => _paramCount;

    private void ValidateIndex(int index)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (index < 0 || index >= _paramCount)
            throw new ArgumentOutOfRangeException(nameof(index),
                $"Parameter index {index} out of range [0, {_paramCount})");
    }

    private static int GetScalarSize(string typeStr)
    {
        return typeStr.ToLowerInvariant() switch
        {
            "float" or "f32" => 4,
            "double" or "f64" => 8,
            "int" or "int32" or "s32" => 4,
            "uint" or "uint32" or "u32" => 4,
            "long" or "int64" or "s64" => 8,
            "ulong" or "uint64" or "u64" => 8,
            "short" or "int16" or "s16" => 2,
            "ushort" or "uint16" or "u16" => 2,
            "byte" or "uint8" or "u8" => 1,
            "sbyte" or "int8" or "s8" => 1,
            _ => 8, // default to 8 bytes (safe for any scalar)
        };
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        for (int i = 0; i < _paramCount; i++)
        {
            if (_paramSlots[i] != IntPtr.Zero)
                Marshal.FreeHGlobal(_paramSlots[i]);
        }

        if (_paramBlock != IntPtr.Zero)
            Marshal.FreeHGlobal(_paramBlock);
    }
}
