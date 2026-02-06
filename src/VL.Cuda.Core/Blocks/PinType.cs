using System;
using VL.Cuda.Core.Buffers;

namespace VL.Cuda.Core.Blocks;

/// <summary>
/// Kind of a block pin: buffer (pointer) or scalar.
/// </summary>
public enum PinKind
{
    Buffer,
    Scalar,
}

/// <summary>
/// Describes the type of a block port: kind (Buffer/Scalar) + data type.
/// Used for connection validation.
/// </summary>
public sealed class PinType : IEquatable<PinType>
{
    public PinKind Kind { get; }
    public DType DataType { get; }

    public PinType(PinKind kind, DType dataType)
    {
        Kind = kind;
        DataType = dataType;
    }

    public static PinType Buffer(DType dataType) => new(PinKind.Buffer, dataType);
    public static PinType Scalar(DType dataType) => new(PinKind.Scalar, dataType);

    public static PinType Buffer<T>() where T : unmanaged
        => new(PinKind.Buffer, DTypeExtensions.FromClrType<T>());

    public static PinType Scalar<T>() where T : unmanaged
        => new(PinKind.Scalar, DTypeExtensions.FromClrType<T>());

    /// <summary>
    /// Whether this pin type is compatible with another for connection.
    /// Both must be the same kind and same data type.
    /// </summary>
    public bool IsCompatible(PinType other)
    {
        if (other is null) return false;
        return Kind == other.Kind && DataType == other.DataType;
    }

    public bool Equals(PinType? other)
    {
        if (other is null) return false;
        return Kind == other.Kind && DataType == other.DataType;
    }

    public override bool Equals(object? obj) => Equals(obj as PinType);
    public override int GetHashCode() => HashCode.Combine(Kind, DataType);
    public override string ToString() => $"{Kind}({DataType})";
}
