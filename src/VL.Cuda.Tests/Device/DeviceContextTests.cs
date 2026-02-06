using VL.Cuda.Core.Device;
using Xunit;

namespace VL.Cuda.Tests.Device;

public class DeviceContextTests
{
    [Fact]
    public void CreateContext_Succeeds()
    {
        using var device = new DeviceContext(0);
        Assert.True(device.DeviceId == 0);
    }

    [Fact]
    public void DeviceName_IsNotEmpty()
    {
        using var device = new DeviceContext(0);
        Assert.False(string.IsNullOrWhiteSpace(device.DeviceName));
    }

    [Fact]
    public void ComputeCapability_IsAtLeast75()
    {
        using var device = new DeviceContext(0);
        var major = device.ComputeCapabilityMajor;
        var minor = device.ComputeCapabilityMinor;
        Assert.True(major > 7 || (major == 7 && minor >= 5),
            $"Compute capability {major}.{minor} is below minimum 7.5");
    }

    [Fact]
    public void TotalMemory_IsPositive()
    {
        using var device = new DeviceContext(0);
        Assert.True(device.TotalMemoryBytes > 0);
    }

    [Fact]
    public void MultiProcessorCount_IsPositive()
    {
        using var device = new DeviceContext(0);
        Assert.True(device.MultiProcessorCount > 0);
    }

    [Fact]
    public void Synchronize_DoesNotThrow()
    {
        using var device = new DeviceContext(0);
        device.Synchronize();
    }
}
