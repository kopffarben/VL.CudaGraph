using Stride.Core.Extensions;
using System;
using System.Collections.Immutable;
using System.Linq;
using VL.Core;
using VL.Core.PublicAPI;
using VL.Lang.PublicAPI;
using VL.Lib.Collections;
using VL.Model;

namespace Main;

[ProcessNode(FragmentSelection = FragmentSelection.Explicit)]
public class PureFunctionBCPInvoker : IDisposable
{
    private readonly NodeContext _nodeContext;
    private readonly UniqueId _nodeId;
    private readonly TypeRegistry _typeRegistry;
    private PureFunctionInterface? _patchInlay;
    private Spread<InputDescription> _lastInputDescs = Spread<InputDescription>.Empty;
    private Spread<OutputDescription> _lastOutputDescs = Spread<OutputDescription>.Empty;
    private Spread<object?> _lastInputValues = Spread<object?>.Empty;
    private readonly List<string> _inputPinNames = new();
    private readonly List<string> _outputPinNames = new();

    private ImmutableDictionary<string, object?> _output = ImmutableDictionary<string, object?>.Empty;
    private int _initFunction;
    private int _nullFunction;

    [Fragment(Order = 0)]
    public PureFunctionBCPInvoker([Pin(Visibility = VL.Model.PinVisibility.Hidden)] NodeContext nodeContext)
    {
        _nodeContext = nodeContext;
        _nodeId = nodeContext.Path.Stack.Peek();
        _typeRegistry = nodeContext.AppHost.TypeRegistry;
    }

    [Fragment(Order = 1)]
    public void Update(
        PureFunctionBCP? function,
        [Pin(Name = "MyInputPinGroup", PinGroupKind = PinGroupKind.Dictionary)] ImmutableDictionary<string, object> input,
        [Pin(Name = "MyOutputPinGroup", PinGroupKind = PinGroupKind.Dictionary)] out ImmutableDictionary<string, object?> output,
        [Pin(Visibility = PinVisibility.Optional)] bool SetDefault = true,
        bool Enable = true)
    {
        output = _output;

        if (function is null)
        {
            HandleNullFunction();
            return;
        }

        _nullFunction = 0;
        _initFunction++;

        if (!Enable)
            return;

        EnsurePatchInlay(function);

        RefreshInputPins(function);
        ApplyDefaultInputs(function, SetDefault);
        CopyInputValuesFromPins(function, input);

        _patchInlay?.Update();

        RefreshOutputPins(function);
        PublishOutputs(function);
    }

    private void HandleNullFunction()
    {
        _nullFunction++;
        _initFunction = 0;
        if (_nullFunction != 1)
            return;

        _lastInputDescs = Spread<InputDescription>.Empty;
        _lastInputValues = Spread<object?>.Empty;
        _lastOutputDescs = Spread<OutputDescription>.Empty;
        _output = ImmutableDictionary<string, object?>.Empty;
    }

    private void EnsurePatchInlay(PureFunctionBCP function)
    {
        _patchInlay ??= function._patchInlayFactory?.Invoke();
    }

    private void RefreshInputPins(PureFunctionBCP function)
    {
        if (_initFunction != 1 && ReferenceEquals(_lastInputDescs, function._inputDescs))
            return;

        _inputPinNames.Clear();
        var pinGroupBuilder = SessionNodes.CurrentSolution.ModifyPinGroup(_nodeId, "MyInputPinGroup", true);
        for (var index = 0; index < function._inputDescs.Count; index++)
        {
            var desc = function._inputDescs[index];
            var name = desc.Name ?? $"Input {index}";
            pinGroupBuilder.Add(name, _typeRegistry.GetTypeInfo(desc.OuterType).FullName);
            _inputPinNames.Add(name);
        }

        pinGroupBuilder.Commit().Confirm(SolutionUpdateKind.TweakLast);
        _lastInputDescs = function._inputDescs;
    }

    private void ApplyDefaultInputs(PureFunctionBCP function, bool setDefault)
    {
        if (!setDefault || ReferenceEquals(_lastInputValues, function._inputValues))
            return;

        var solution = SessionNodes.CurrentSolution;
        var values = function._inputValues;
        for (var i = 0; i < values.Count; i++)
        {
            solution.SetPinValue(_nodeId, _inputPinNames[i], values[i]);
        }

        solution.Confirm(SolutionUpdateKind.DontCompile);
        _lastInputValues = values;
    }

    private static Spread<object?> CopyInputValuesFromPins(PureFunctionBCP function, ImmutableDictionary<string, object> input, IReadOnlyList<string> inputPinNames)
    {
        var builder = function._inputValuesFromInvoke.ToSpreadBuilder();
        builder.Clear();
        foreach (var name in inputPinNames)
        {
            builder.Add(input[name]);
        }

        return builder.ToSpread();
    }

    private void CopyInputValuesFromPins(PureFunctionBCP function, ImmutableDictionary<string, object> input)
    {
        function._inputValuesFromInvoke = CopyInputValuesFromPins(function, input, _inputPinNames);
    }

    private void RefreshOutputPins(PureFunctionBCP function)
    {
        if (ReferenceEquals(_lastOutputDescs, function._outputDescs))
            return;

        _outputPinNames.Clear();
        var outputBuilder = _output.ToBuilder();
        outputBuilder.Clear();
        var pinGroupBuilder = SessionNodes.CurrentSolution.ModifyPinGroup(_nodeId, "MyOutputPinGroup", false);

        for (var i = 0; i < function._outputDescs.Count; i++)
        {
            var desc = function._outputDescs[i];
            var val = function._outputValues[i];
            var name = desc.Name ?? $"Output {i}";
            pinGroupBuilder.Add(name, _typeRegistry.GetTypeInfo(desc.OuterType).FullName);
            _outputPinNames.Add(name);
            outputBuilder.Add(name, val);
        }

        pinGroupBuilder.Commit().Confirm(SolutionUpdateKind.TweakLast);
        _lastOutputDescs = function._outputDescs;
        _output = outputBuilder.ToImmutable();
    }

    private void PublishOutputs(PureFunctionBCP function)
    {
        var outputBuilder = _output.ToBuilder();
        var changed = false;

        for (var i = 0; i < function._outputValues.Count; i++)
        {
            var val = function._outputValues[i];
            var pinName = _outputPinNames[i];
            if (outputBuilder[pinName] == val)
                continue;

            outputBuilder[pinName] = val;
            changed = true;
        }

        if (changed)
            _output = outputBuilder.ToImmutable();
    }

    [Fragment(Order = 2)]
    public void Dispose()
    {
        if (_patchInlay is IDisposable disposable)
        {
            disposable.Dispose();
        }
    }
}

[ProcessNode(FragmentSelection = FragmentSelection.Explicit, HasStateOutput = true)]
[Region(SupportedBorderControlPoints = ControlPointType.Border)]
public class PureFunctionBCP : IRegion<PureFunctionInterface>
{
    private NodeContext _nodeContext;
    internal Func<PureFunctionInterface>? _patchInlayFactory;
    internal Spread<InputDescription> _inputDescs = Spread<InputDescription>.Empty;
    internal Spread<object?> _inputValues = Spread<object?>.Empty;
    internal Spread<object?> _inputValuesFromInvoke = Spread<object?>.Empty;
    internal Spread<OutputDescription> _outputDescs = Spread<OutputDescription>.Empty;
    internal Spread<object?> _outputValues = Spread<object?>.Empty;
    private readonly HashSet<InputDescription> _seenInputs = new();
    private readonly HashSet<OutputDescription> _seenOutputs = new();
    private int _inputIndex;
    private int _outputIndex;

    [Fragment(Order = 0)]
    public PureFunctionBCP([Pin(Visibility = VL.Model.PinVisibility.Hidden)] NodeContext nodeContext)
    {
        _nodeContext = nodeContext;
    }

    [Fragment(Order = 1)]
    public void Update()
    {
        PruneStaleEntries();
        _inputIndex = 0;
        _outputIndex = 0;
    }

    #region IRegion<TestReginInterface> Members
    public void AcknowledgeInput(in InputDescription description, object? outerValue)
    {
        _seenInputs.Add(description);
        var existingIndex = _inputDescs.IndexOf(description);

        if (existingIndex < 0)
        {
            if (_inputIndex < _inputDescs.Count)
            {
                var db = _inputDescs.ToSpreadBuilder();
                var vb = _inputValues.ToSpreadBuilder();
                db.SetRange([description], _inputIndex);
                vb.SetRange([outerValue], _inputIndex);
                _inputDescs = db.ToSpread();
                _inputValues = vb.ToSpread();
            }
            else
            {
                var db = _inputDescs.ToSpreadBuilder();
                var vb = _inputValues.ToSpreadBuilder();
                db.Add(description);
                vb.Add(outerValue);
                _inputDescs = db.ToSpread();
                _inputValues = vb.ToSpread();
            }
        }
        else if (!Equals(_inputValues[existingIndex], outerValue))
        {
            var vb = _inputValues.ToSpreadBuilder();
            vb.SetRange([outerValue], existingIndex);
            _inputValues = vb.ToSpread();
        }

        _inputIndex++;
    }

    public void AcknowledgeOutput(in OutputDescription description, PureFunctionInterface patchInlay, object? innerValue)
    {
        _seenOutputs.Add(description);
        var existingIndex = _outputDescs.IndexOf(description);

        if (existingIndex < 0)
        {
            if (_outputIndex < _outputDescs.Count)
            {
                var db = _outputDescs.ToSpreadBuilder();
                var vb = _outputValues.ToSpreadBuilder();
                db.SetRange([description], _outputIndex);
                vb.SetRange([innerValue], _outputIndex);
                _outputDescs = db.ToSpread();
                _outputValues = vb.ToSpread();
            }
            else
            {
                var db = _outputDescs.ToSpreadBuilder();
                var vb = _outputValues.ToSpreadBuilder();
                db.Add(description);
                vb.Add(innerValue);
                _outputDescs = db.ToSpread();
                _outputValues = vb.ToSpread();
            }
        }
        else if (!Equals(_outputValues[existingIndex], innerValue))
        {
            var vb = _outputValues.ToSpreadBuilder();
            vb.SetRange([innerValue], existingIndex);
            _outputValues = vb.ToSpread();
        }

        _outputIndex++;
    }

    public void RetrieveInput(in InputDescription description, PureFunctionInterface patchInlay, out object? innerValue)
    {
        var existingIndex = _inputDescs.IndexOf(description);
        if (existingIndex < 0) innerValue = null;
        else
            if (existingIndex < _inputValuesFromInvoke.Count)
                innerValue = _inputValuesFromInvoke[existingIndex];
            else
                innerValue = null;
    }

    public void RetrieveOutput(in OutputDescription description, out object? outerValue)
    {
        var existingIndex = _outputDescs.IndexOf(description);
        if (existingIndex < 0) outerValue = null;
        else
            if (existingIndex < _outputValues.Count)
                outerValue = _outputValues[existingIndex];
            else
                outerValue = null;
    }

    public void SetPatchInlayFactory(Func<PureFunctionInterface> patchInlayFactory)
    {
        this._patchInlayFactory = patchInlayFactory;
    }
    #endregion IRegion<TestReginInterface> Members

    internal void PruneStaleEntries()
    {
        if (_inputDescs.Count > 0)
        {
            var staleIndices = Enumerable.Range(0, _inputDescs.Count)
                .Where(i => !_seenInputs.Contains(_inputDescs[i]))
                .ToArray();
            if (staleIndices.Length > 0)
            {
                var db = _inputDescs.ToSpreadBuilder();
                var vb = _inputValues.ToSpreadBuilder();
                for (var i = staleIndices.Length - 1; i >= 0; i--)
                {
                    var idx = staleIndices[i];
                    db.RemoveAt(idx);
                    vb.RemoveAt(idx);
                }
                _inputDescs = db.ToSpread();
                _inputValues = vb.ToSpread();
            }
        }

        if (_outputDescs.Count > 0)
        {
            var staleIndices = Enumerable.Range(0, _outputDescs.Count)
                .Where(i => !_seenOutputs.Contains(_outputDescs[i]))
                .ToArray();
            if (staleIndices.Length > 0)
            {
                var db = _outputDescs.ToSpreadBuilder();
                var vb = _outputValues.ToSpreadBuilder();
                for (var i = staleIndices.Length - 1; i >= 0; i--)
                {
                    var idx = staleIndices[i];
                    db.RemoveAt(idx);
                    vb.RemoveAt(idx);
                }
                _outputDescs = db.ToSpread();
                _outputValues = vb.ToSpread();
            }
        }

        _seenInputs.Clear();
        _seenOutputs.Clear();
    }
}