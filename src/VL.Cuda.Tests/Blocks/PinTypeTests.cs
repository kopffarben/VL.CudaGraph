using VL.Cuda.Core.Blocks;
using VL.Cuda.Core.Buffers;
using Xunit;

namespace VL.Cuda.Tests.Blocks;

public class PinTypeTests
{
    [Fact]
    public void Buffer_Float_IsCompatible_WithSame()
    {
        var a = PinType.Buffer(DType.F32);
        var b = PinType.Buffer(DType.F32);
        Assert.True(a.IsCompatible(b));
    }

    [Fact]
    public void Buffer_Float_NotCompatible_WithDouble()
    {
        var a = PinType.Buffer(DType.F32);
        var b = PinType.Buffer(DType.F64);
        Assert.False(a.IsCompatible(b));
    }

    [Fact]
    public void Buffer_NotCompatible_WithScalar()
    {
        var a = PinType.Buffer(DType.F32);
        var b = PinType.Scalar(DType.F32);
        Assert.False(a.IsCompatible(b));
    }

    [Fact]
    public void Scalar_Int_IsCompatible_WithSame()
    {
        var a = PinType.Scalar(DType.S32);
        var b = PinType.Scalar(DType.S32);
        Assert.True(a.IsCompatible(b));
    }

    [Fact]
    public void Generic_Buffer_Float()
    {
        var pin = PinType.Buffer<float>();
        Assert.Equal(PinKind.Buffer, pin.Kind);
        Assert.Equal(DType.F32, pin.DataType);
    }

    [Fact]
    public void Generic_Scalar_UInt()
    {
        var pin = PinType.Scalar<uint>();
        Assert.Equal(PinKind.Scalar, pin.Kind);
        Assert.Equal(DType.U32, pin.DataType);
    }

    [Fact]
    public void IsCompatible_Null_ReturnsFalse()
    {
        var a = PinType.Buffer(DType.F32);
        Assert.False(a.IsCompatible(null!));
    }

    [Fact]
    public void Equals_SameValues_True()
    {
        var a = PinType.Buffer(DType.F32);
        var b = PinType.Buffer(DType.F32);
        Assert.True(a.Equals(b));
        Assert.Equal(a.GetHashCode(), b.GetHashCode());
    }

    [Fact]
    public void Equals_DifferentValues_False()
    {
        var a = PinType.Buffer(DType.F32);
        var b = PinType.Buffer(DType.F64);
        Assert.False(a.Equals(b));
    }

    [Fact]
    public void ToString_DescriptiveFormat()
    {
        var pin = PinType.Buffer(DType.F32);
        Assert.Contains("Buffer", pin.ToString());
        Assert.Contains("F32", pin.ToString());
    }
}
