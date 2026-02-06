using System;
using System.Collections.Generic;
using VL.Core;
using VL.Cuda.Core.Blocks;
using VL.Cuda.Core.Blocks.Builder;
using VL.Cuda.Core.Context;

namespace VL.Cuda.Tests.Helpers;

/// <summary>
/// Simple ICudaBlock implementation for testing. Allows manual control
/// of all properties without requiring actual PTX files.
/// </summary>
public sealed class TestBlock : ICudaBlock
{
    private readonly List<IBlockPort> _inputs = new();
    private readonly List<IBlockPort> _outputs = new();
    private readonly List<IBlockParameter> _parameters = new();

    public Guid Id { get; }
    public string TypeName { get; }
    public NodeContext NodeContext { get; }

    public IReadOnlyList<IBlockPort> Inputs => _inputs;
    public IReadOnlyList<IBlockPort> Outputs => _outputs;
    public IReadOnlyList<IBlockParameter> Parameters => _parameters;
    public IBlockDebugInfo DebugInfo { get; set; } = new BlockDebugInfo();

    public TestBlock(string typeName = "TestBlock")
    {
        Id = Guid.NewGuid();
        TypeName = typeName;
        NodeContext = TestNodeContext.Create();
    }

    public TestBlock(Guid id, string typeName = "TestBlock")
    {
        Id = id;
        TypeName = typeName;
        NodeContext = TestNodeContext.Create();
    }

    public void AddInput(BlockPort port) => _inputs.Add(port);
    public void AddOutput(BlockPort port) => _outputs.Add(port);
    public void AddParameter(IBlockParameter param) => _parameters.Add(param);

    public void Dispose() { }
}

/// <summary>
/// TestBlock that is built using BlockBuilder against a CudaContext.
/// This requires actual PTX files but tests the full builder pipeline.
/// </summary>
public sealed class BuiltTestBlock : ICudaBlock
{
    private readonly CudaContext _ctx;
    private readonly List<IBlockPort> _inputs = new();
    private readonly List<IBlockPort> _outputs = new();
    private readonly List<IBlockParameter> _parameters = new();

    public Guid Id { get; } = Guid.NewGuid();
    public string TypeName { get; }
    public NodeContext NodeContext { get; }
    public IReadOnlyList<IBlockPort> Inputs => _inputs;
    public IReadOnlyList<IBlockPort> Outputs => _outputs;
    public IReadOnlyList<IBlockParameter> Parameters => _parameters;
    public IBlockDebugInfo DebugInfo { get; set; } = new BlockDebugInfo();

    /// <summary>
    /// The block builder for external access to kernel handles, ports etc.
    /// </summary>
    public BlockBuilder Builder { get; private set; } = null!;

    public BuiltTestBlock(CudaContext ctx, string ptxPath, string typeName = "BuiltTestBlock",
        Action<BlockBuilder, BuiltTestBlock>? configure = null)
    {
        _ctx = ctx;
        TypeName = typeName;
        NodeContext = TestNodeContext.Create();

        var builder = new BlockBuilder(ctx, this);
        Builder = builder;

        if (configure != null)
        {
            configure(builder, this);
        }
        else
        {
            // Default: load the PTX and expose all ports
            var kernel = builder.AddKernel(ptxPath);

            foreach (var param in kernel.Descriptor.Parameters)
            {
                if (param.IsPointer)
                {
                    if (param.Direction == VL.Cuda.Core.PTX.ParamDirection.Out ||
                        param.Direction == VL.Cuda.Core.PTX.ParamDirection.InOut)
                    {
                        var port = builder.Output<float>(param.Name, kernel.Out(param.Index));
                        _outputs.Add(port);
                    }
                    else
                    {
                        var port = builder.Input<float>(param.Name, kernel.In(param.Index));
                        _inputs.Add(port);
                    }
                }
                else
                {
                    if (param.Type.ToLowerInvariant().Contains("float"))
                    {
                        var bp = builder.InputScalar<float>(param.Name, kernel.In(param.Index));
                        _parameters.Add(bp);
                    }
                    else
                    {
                        var bp = builder.InputScalar<uint>(param.Name, kernel.In(param.Index));
                        _parameters.Add(bp);
                    }
                }
            }
        }

        builder.Commit();
        ctx.RegisterBlock(this);
    }

    public void SetInputs(List<IBlockPort> inputs) => _inputs.AddRange(inputs);
    public void SetOutputs(List<IBlockPort> outputs) => _outputs.AddRange(outputs);
    public void SetParameters(List<IBlockParameter> parameters) => _parameters.AddRange(parameters);

    public void Dispose()
    {
        _ctx.UnregisterBlock(Id);
    }
}
