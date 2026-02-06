using System;
using System.Runtime.InteropServices;

namespace VL.Cuda.Core.Buffers;

/// <summary>
/// GPU data types with known sizes. Used for type-erased buffer operations.
/// </summary>
public enum DType
{
    F16 = 0,
    F32 = 1,
    F64 = 2,
    S8 = 3,
    S16 = 4,
    S32 = 5,
    S64 = 6,
    U8 = 7,
    U16 = 8,
    U32 = 9,
    U64 = 10,
}

public static class DTypeExtensions
{
    public static int SizeInBytes(this DType dtype) => dtype switch
    {
        DType.F16 => 2,
        DType.F32 => 4,
        DType.F64 => 8,
        DType.S8 => 1,
        DType.S16 => 2,
        DType.S32 => 4,
        DType.S64 => 8,
        DType.U8 => 1,
        DType.U16 => 2,
        DType.U32 => 4,
        DType.U64 => 8,
        _ => throw new ArgumentOutOfRangeException(nameof(dtype))
    };

    public static DType FromClrType<T>() where T : unmanaged
    {
        if (typeof(T) == typeof(float)) return DType.F32;
        if (typeof(T) == typeof(double)) return DType.F64;
        if (typeof(T) == typeof(sbyte)) return DType.S8;
        if (typeof(T) == typeof(short)) return DType.S16;
        if (typeof(T) == typeof(int)) return DType.S32;
        if (typeof(T) == typeof(long)) return DType.S64;
        if (typeof(T) == typeof(byte)) return DType.U8;
        if (typeof(T) == typeof(ushort)) return DType.U16;
        if (typeof(T) == typeof(uint)) return DType.U32;
        if (typeof(T) == typeof(ulong)) return DType.U64;
        if (typeof(T) == typeof(Half)) return DType.F16;
        throw new NotSupportedException($"No DType mapping for {typeof(T).Name}");
    }

    public static int SizeOf<T>() where T : unmanaged => Marshal.SizeOf<T>();
}
