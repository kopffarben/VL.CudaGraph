using System;

namespace VL.Cuda.Core.Buffers;

/// <summary>
/// Describes the multi-dimensional layout of a GPU buffer.
/// Phase 0: 1D only. Multi-dimensional support added later.
/// </summary>
public readonly struct BufferShape : IEquatable<BufferShape>
{
    public int X { get; }
    public int Y { get; }
    public int Z { get; }
    public int Rank { get; }

    /// <summary>Total number of elements across all dimensions.</summary>
    public long TotalElements => (long)X * Y * Z;

    private BufferShape(int x, int y, int z, int rank)
    {
        X = x;
        Y = y;
        Z = z;
        Rank = rank;
    }

    public static BufferShape D1(int count) => new(count, 1, 1, 1);
    public static BufferShape D2(int width, int height) => new(width, height, 1, 2);
    public static BufferShape D3(int x, int y, int z) => new(x, y, z, 3);

    public bool Equals(BufferShape other) => X == other.X && Y == other.Y && Z == other.Z;
    public override bool Equals(object? obj) => obj is BufferShape other && Equals(other);
    public override int GetHashCode() => HashCode.Combine(X, Y, Z);
    public override string ToString() => Rank switch
    {
        1 => $"[{X}]",
        2 => $"[{X}x{Y}]",
        _ => $"[{X}x{Y}x{Z}]"
    };

    public static bool operator ==(BufferShape left, BufferShape right) => left.Equals(right);
    public static bool operator !=(BufferShape left, BufferShape right) => !left.Equals(right);
}
