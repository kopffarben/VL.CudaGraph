using VL.Core.PublicAPI;


namespace Main;

[ProcessNode( FragmentSelection = FragmentSelection.Explicit)]
[Region(SupportedBorderControlPoints = ControlPointType.Splicer , TypeConstraint = "Spread", TypeConstraintIsBaseType = true)]
public class ForeachRegion : IRegion<ForeachRegionInterface>, IDisposable
{
    private NodeContext _nodeContext;
    private ForeachRegionInterface? _patchInlay;
    private Func<ForeachRegionInterface>? _patchInlayFactory;
    private readonly Dictionary<InputDescription,  ISpread> _inputSplicers  = new();
    private readonly Dictionary<OutputDescription, ISpreadBuilder> _outputSplicers = new();
    private int _currentIndex = 0;
    private int _iterationCount = 1;


    [Fragment(Order = 0)]
    public ForeachRegion( [Pin(Visibility = VL.Model.PinVisibility.Hidden)] NodeContext nodeContext )
    {
        _nodeContext = nodeContext;
    }


    [Fragment(Order = 1)]
    public void Update()
    {
        if (_patchInlayFactory != null && _patchInlay == null)
        {
            this._patchInlay = _patchInlayFactory();
        }
        if (this._patchInlay != null)
        {
            for (_currentIndex = 0; _currentIndex < _iterationCount; _currentIndex++)
            {
                _patchInlay.Update(_currentIndex, out var _break);
                if (_break) break;
            }
        }    
    }

    /// <inheritdoc />
    [Fragment(Order = 2)]
    public void Dispose()
    {
        if (_patchInlay is IDisposable disposable)
        {
            disposable.Dispose();
        }
    }

    #region IRegion<TestReginInterface> Members
    /// <inheritdoc />
    public void AcknowledgeInput(in InputDescription description, object? outerValue)
    {
        if (outerValue is ISpread spread)
        {
            _iterationCount = spread.Count;
            if (!_inputSplicers.ContainsValue(spread))
                _inputSplicers[description] = spread;
        }
    }

    /// <inheritdoc />
    public void RetrieveInput(in InputDescription description, ForeachRegionInterface patchInlay, out object? innerValue)
    {
        innerValue = _inputSplicers[description]!.GetItem(_currentIndex);
    }
    /// <inheritdoc />
    public void AcknowledgeOutput(in OutputDescription description, ForeachRegionInterface patchInlay, object? innerValue)
    {  
        if (!_outputSplicers.ContainsKey(description))
        {
            _outputSplicers[description] = Helper.CreateSpreadBuilder(innerValue!, _iterationCount);
        }
        else if (_outputSplicers[description].Count <= _currentIndex)
        {
            _outputSplicers[description].Add(innerValue);
        }
    }

    /// <inheritdoc />
    public void RetrieveOutput(in OutputDescription description, out object? outerValue)
    {
                  
        outerValue = _outputSplicers[description]!.ToSpread();
                     _outputSplicers[description].Clear();
    }

    /// <summary>
    /// Sets the factory function used to create the patch inlay instance.
    /// </summary>
    /// <param name="patchInlayFactory">The factory function.</param>
    public void SetPatchInlayFactory(Func<ForeachRegionInterface> patchInlayFactory)
    {
        this._patchInlayFactory = patchInlayFactory;
    }
    #endregion IRegion<TestReginInterface> Members

}