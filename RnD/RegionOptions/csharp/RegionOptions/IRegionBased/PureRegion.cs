using VL.AppServices.CompilerServices.CustomRegion;
using VL.AppServices.Hotswap;
using VL.Core.PublicAPI;

namespace Main;

/// <summary>
/// Represents a pure region handling LFO interface logic.
/// </summary>
[ProcessNode( FragmentSelection = FragmentSelection.Explicit)]
[Region(SupportedBorderControlPoints = ControlPointType.None)]
public class PureRegion : IRegion<LFOInterface>, IDisposable
{
    private NodeContext _nodeContext;
    private LFOInterface? _patchInlay;
    private Func<LFOInterface>? _patchInlayFactory;

    /// <summary>
    /// Initializes a new instance of the <see cref="PureRegion"/> class.
    /// </summary>
    /// <param name="nodeContext">The node context provided by the system.</param>
    [Fragment(Order = 0)]
    public PureRegion( [Pin(Visibility = VL.Model.PinVisibility.Hidden)] NodeContext nodeContext )
    {
        _nodeContext = nodeContext;
    }

    /// <summary>
    /// Updates the region logic using the patch inlay.
    /// </summary>
    /// <param name="Period">The period for the LFO.</param>
    /// <param name="Pause">Whether the LFO should pause.</param>
    /// <param name="Reset">Whether the LFO should reset.</param>
    /// <param name="Result">The output value of the LFO.</param>
    [Fragment(Order = 1)]
    public void Update(float Period, bool Pause, bool Reset, out float Result)
    {
        if (_patchInlayFactory != null && _patchInlay == null)
        {
            this._patchInlay = _patchInlayFactory();
        }
        if (this._patchInlay != null)
        {
            this._patchInlay.Update(Period, Pause, Reset, out Result);
        }
        else
        {
            Result = default;
        }
    }

    #region IDisposable Members
    /// <inheritdoc />
    [Fragment(Order = 2)]
    public void Dispose()
    {
        if (_patchInlay is IDisposable disposable)
        {
            disposable.Dispose();
        }
    }
    #endregion IDisposable Members

    #region IRegion<TestReginInterface> Members
    /// <inheritdoc />
    public void AcknowledgeInput(in InputDescription description, object? outerValue)
    {
        // no BorderControlPoints are supported, so this method will never be called
    }

    /// <inheritdoc />
    public void AcknowledgeOutput(in OutputDescription description, LFOInterface patchInlay, object? innerValue)
    {
        // no BorderControlPoints are supported, so this method will never be called
    }

    /// <inheritdoc />
    public void RetrieveInput(in InputDescription description, LFOInterface patchInlay, out object? innerValue)
    {
        // no BorderControlPoints are supported, so this method will never be called
        innerValue = null;
    }

    /// <inheritdoc />
    public void RetrieveOutput(in OutputDescription description, out object? outerValue)
    {
        // no BorderControlPoints are supported, so this method will never be called
        outerValue = null;
    }

    /// <summary>
    /// Sets the factory function used to create the patch inlay instance.
    /// </summary>
    /// <param name="patchInlayFactory">The factory function.</param>
    public void SetPatchInlayFactory(Func<LFOInterface> patchInlayFactory)
    {
        this._patchInlayFactory = patchInlayFactory;
    }
    #endregion IRegion<TestReginInterface> Members
   
}