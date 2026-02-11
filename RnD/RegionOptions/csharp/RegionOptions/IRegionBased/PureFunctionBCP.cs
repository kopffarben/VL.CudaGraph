using System.Collections.Immutable;
using VL.Core;
using VL.Core.PublicAPI;
using VL.Lang.PublicAPI;
using VL.Model;

namespace Main;

[ProcessNode(FragmentSelection = FragmentSelection.Explicit)]
public class PureFunctionBCPInvoker : IDisposable
{
    private NodeContext _nodeContext;
    private readonly UniqueId _nodeId;
    private TypeRegistry _typeRegistry;
    private PureFunctionInterface? _patchInlay;

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
        bool SetDefault = false,
        bool Reset = false,
        bool Enable = true)
    {
        if (function == null) return;
        
        if (Enable)
        {
            if (_patchInlay == null)
            {
                _patchInlay = function._patchInlayFactory?.Invoke();
            }
            else
            {
                int index = 0;
                if (Reset)
                {
                    var pingroupBuilder = SessionNodes.CurrentSolution.ModifyPinGroup(_nodeId, "MyInputPinGroup", true);
                    foreach (InputDescription description in function._inputValues.Keys)
                    {
                        pingroupBuilder.Add(description.Name == null ? "Input_" + index.ToString() : description.Name, _typeRegistry.GetTypeInfo(description.OuterType).FullName);
                        index++;
                    }
                    pingroupBuilder.Commit().Confirm(SolutionUpdateKind.TweakLast);
                    index = 0;
                }
                if (SetDefault)
                {
                    var solution = SessionNodes.CurrentSolution;
                    foreach (var kvp in function._inputValues)
                    {
                        solution.SetPinValue(_nodeId, kvp.Key.Name == null ? "Input_" + index.ToString() : kvp.Key.Name, kvp.Value);
                        index++;
                    }
                    solution.Confirm(SolutionUpdateKind.DontCompile);
                }
            }
        }
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
    public Func<PureFunctionInterface>? _patchInlayFactory;
    public readonly Dictionary<InputDescription,  object> _inputValues  = new Dictionary<InputDescription,  object>();
    public readonly Dictionary<OutputDescription, object> _outputValues = new Dictionary<OutputDescription, object>();

    [Fragment(Order = 0)]
    public PureFunctionBCP([Pin(Visibility = VL.Model.PinVisibility.Hidden)] NodeContext nodeContext)
    {
        _nodeContext = nodeContext;
    }

    [Fragment(Order = 1)]
    public void Update()
    {
        
    }

    #region IRegion<TestReginInterface> Members
    public void AcknowledgeInput(in InputDescription description, object? outerValue)
    {
        _inputValues[description] = outerValue!;
    }

    public void AcknowledgeOutput(in OutputDescription description, PureFunctionInterface patchInlay, object? innerValue)
    {
        _outputValues[description] = innerValue!;
    }

    public void RetrieveInput(in InputDescription description, PureFunctionInterface patchInlay, out object? innerValue)
    {
        // no BorderControlPoints are supported, so this method will never be called
        innerValue = null;
    }

    public void RetrieveOutput(in OutputDescription description, out object? outerValue)
    {
        // no BorderControlPoints are supported, so this method will never be called
        outerValue = null;
    }

    public void SetPatchInlayFactory(Func<PureFunctionInterface> patchInlayFactory)
    {
        this._patchInlayFactory = patchInlayFactory;
    }
    #endregion IRegion<TestReginInterface> Members

}