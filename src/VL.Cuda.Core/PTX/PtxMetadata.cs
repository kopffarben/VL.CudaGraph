using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace VL.Cuda.Core.PTX;

/// <summary>
/// Parses the companion .json file for a .ptx kernel.
/// </summary>
public static class PtxMetadata
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    /// <summary>
    /// Parse a kernel metadata JSON file.
    /// </summary>
    public static KernelDescriptor Parse(string jsonPath)
    {
        var json = File.ReadAllText(jsonPath);
        return ParseJson(json);
    }

    /// <summary>
    /// Parse kernel metadata from a JSON string.
    /// </summary>
    public static KernelDescriptor ParseJson(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        var entryPoint = root.GetProperty("entryPoint").GetString()
            ?? throw new InvalidOperationException("entryPoint is required");

        var parameters = new List<KernelParamDescriptor>();
        if (root.TryGetProperty("parameters", out var paramsElement))
        {
            int index = 0;
            foreach (var p in paramsElement.EnumerateArray())
            {
                var name = p.GetProperty("name").GetString()
                    ?? throw new InvalidOperationException("parameter name is required");
                var type = p.GetProperty("type").GetString()
                    ?? throw new InvalidOperationException("parameter type is required");

                var direction = ParamDirection.In;
                if (p.TryGetProperty("direction", out var dirElem))
                {
                    direction = dirElem.GetString()?.ToLowerInvariant() switch
                    {
                        "out" => ParamDirection.Out,
                        "inout" => ParamDirection.InOut,
                        _ => ParamDirection.In,
                    };
                }

                var isPointer = false;
                if (p.TryGetProperty("isPointer", out var ptrElem))
                {
                    isPointer = ptrElem.GetBoolean();
                }

                parameters.Add(new KernelParamDescriptor
                {
                    Name = name,
                    Type = type,
                    Index = index++,
                    Direction = direction,
                    IsPointer = isPointer,
                });
            }
        }

        int blockSize = 0;
        if (root.TryGetProperty("blockSize", out var bsElem))
            blockSize = bsElem.GetInt32();

        int sharedMem = 0;
        if (root.TryGetProperty("sharedMemoryBytes", out var smElem))
            sharedMem = smElem.GetInt32();

        return new KernelDescriptor
        {
            EntryPoint = entryPoint,
            Parameters = parameters,
            BlockSize = blockSize,
            SharedMemoryBytes = sharedMem,
        };
    }
}
