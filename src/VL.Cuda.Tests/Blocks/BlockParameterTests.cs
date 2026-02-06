using System;
using VL.Cuda.Core.Blocks;
using Xunit;

namespace VL.Cuda.Tests.Blocks;

public class BlockParameterTests
{
    [Fact]
    public void DefaultValue_IsDefault()
    {
        var param = new BlockParameter<float>("Scale");
        Assert.Equal(0f, param.TypedValue);
    }

    [Fact]
    public void CustomDefault_IsUsed()
    {
        var param = new BlockParameter<float>("Scale", 9.81f);
        Assert.Equal(9.81f, param.TypedValue);
    }

    [Fact]
    public void SetValue_UpdatesTypedValue()
    {
        var param = new BlockParameter<float>("Scale", 1.0f);
        param.TypedValue = 2.5f;
        Assert.Equal(2.5f, param.TypedValue);
    }

    [Fact]
    public void SetValue_MarksDirty()
    {
        var param = new BlockParameter<float>("Scale", 1.0f);
        Assert.False(param.IsDirty);

        param.TypedValue = 2.0f;

        Assert.True(param.IsDirty);
    }

    [Fact]
    public void SetSameValue_DoesNotMarkDirty()
    {
        var param = new BlockParameter<float>("Scale", 1.0f);
        param.TypedValue = 1.0f;

        Assert.False(param.IsDirty);
    }

    [Fact]
    public void ClearDirty_ResetsDirtyFlag()
    {
        var param = new BlockParameter<float>("Scale", 1.0f);
        param.TypedValue = 2.0f;
        Assert.True(param.IsDirty);

        param.ClearDirty();

        Assert.False(param.IsDirty);
    }

    [Fact]
    public void ValueChanged_FiresOnChange()
    {
        var param = new BlockParameter<float>("Scale", 1.0f);
        int fireCount = 0;
        param.ValueChanged += _ => fireCount++;

        param.TypedValue = 2.0f;

        Assert.Equal(1, fireCount);
    }

    [Fact]
    public void ValueChanged_DoesNotFireOnSameValue()
    {
        var param = new BlockParameter<float>("Scale", 1.0f);
        int fireCount = 0;
        param.ValueChanged += _ => fireCount++;

        param.TypedValue = 1.0f;

        Assert.Equal(0, fireCount);
    }

    [Fact]
    public void IBlockParameter_Value_GetSet()
    {
        IBlockParameter param = new BlockParameter<float>("Scale", 1.0f);
        Assert.Equal(1.0f, (float)param.Value);

        param.Value = 3.0f;
        Assert.Equal(3.0f, (float)param.Value);
    }

    [Fact]
    public void ValueType_IsCorrect()
    {
        var param = new BlockParameter<uint>("Count");
        Assert.Equal(typeof(uint), param.ValueType);
    }

    [Fact]
    public void Name_IsCorrect()
    {
        var param = new BlockParameter<float>("Gravity");
        Assert.Equal("Gravity", param.Name);
    }

    [Fact]
    public void IntParameter_Works()
    {
        var param = new BlockParameter<int>("Mode", 42);
        Assert.Equal(42, param.TypedValue);
        param.TypedValue = 99;
        Assert.Equal(99, param.TypedValue);
        Assert.True(param.IsDirty);
    }

    [Fact]
    public void ToString_ShowsNameAndValue()
    {
        var param = new BlockParameter<float>("Scale", 2.5f);
        Assert.Contains("Scale", param.ToString());
    }
}
