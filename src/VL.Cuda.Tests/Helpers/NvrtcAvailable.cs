using System;
using System.Runtime.InteropServices;

namespace VL.Cuda.Tests.Helpers;

/// <summary>
/// Skip tests when the NVRTC native DLL is not available.
/// NVRTC requires the CUDA toolkit to be installed (not just the driver).
/// </summary>
public static class NvrtcAvailable
{
    private static readonly Lazy<bool> _isAvailable = new(() =>
    {
        try
        {
            // Try to load the NVRTC DLL
            NativeLibrary.TryLoad("nvrtc64_130_0", out var handle);
            if (handle != IntPtr.Zero)
            {
                NativeLibrary.Free(handle);
                return true;
            }
            return false;
        }
        catch
        {
            return false;
        }
    });

    public static bool IsAvailable => _isAvailable.Value;

    public static string SkipReason => "NVRTC DLL (nvrtc64_130_0) not found. Install CUDA Toolkit 13.0+.";
}
