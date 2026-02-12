using System.Collections.Generic;

namespace VL.Cuda.Core.Context.Services;

/// <summary>
/// Tracks dirty state for the CUDA pipeline. Subscribes to StructureChanged
/// events from BlockRegistry and ConnectionGraph. CudaEngine reads these
/// flags each frame to decide: Cold Rebuild vs Recapture vs Hot/Warm Update vs no-op.
/// </summary>
public sealed class DirtyTracker
{
    private readonly HashSet<DirtyParameter> _dirtyParameters = new();
    private readonly HashSet<DirtyCapturedNode> _dirtyCapturedNodes = new();

    public bool IsStructureDirty { get; private set; } = true; // Start dirty → first build
    public bool AreParametersDirty => _dirtyParameters.Count > 0;
    public bool AreCapturedNodesDirty => _dirtyCapturedNodes.Count > 0;

    /// <summary>
    /// Subscribe to BlockRegistry and ConnectionGraph events.
    /// </summary>
    public void Subscribe(BlockRegistry registry, ConnectionGraph connectionGraph)
    {
        registry.StructureChanged += OnStructureChanged;
        connectionGraph.StructureChanged += OnStructureChanged;
    }

    private void OnStructureChanged()
    {
        IsStructureDirty = true;
    }

    /// <summary>
    /// Mark a specific parameter as dirty (called by CudaContext.OnParameterChanged).
    /// </summary>
    public void MarkParameterDirty(DirtyParameter param)
    {
        _dirtyParameters.Add(param);
    }

    /// <summary>
    /// Mark a captured node as needing recapture (parameter or configuration changed).
    /// </summary>
    public void MarkCapturedNodeDirty(DirtyCapturedNode node)
    {
        _dirtyCapturedNodes.Add(node);
    }

    /// <summary>
    /// Get all dirty parameters.
    /// </summary>
    public IReadOnlySet<DirtyParameter> GetDirtyParameters()
    {
        return _dirtyParameters;
    }

    /// <summary>
    /// Get all captured nodes needing recapture.
    /// </summary>
    public IReadOnlySet<DirtyCapturedNode> GetDirtyCapturedNodes()
    {
        return _dirtyCapturedNodes;
    }

    /// <summary>
    /// Clear structure dirty flag after Cold Rebuild.
    /// Also clears parameter and recapture dirty since rebuild applies all current values.
    /// </summary>
    public void ClearStructureDirty()
    {
        IsStructureDirty = false;
        _dirtyParameters.Clear();
        _dirtyCapturedNodes.Clear();
    }

    /// <summary>
    /// Clear parameter dirty flags after Hot/Warm Update.
    /// </summary>
    public void ClearParametersDirty()
    {
        _dirtyParameters.Clear();
    }

    /// <summary>
    /// Clear recapture dirty flags after Recapture Update.
    /// </summary>
    public void ClearCapturedNodesDirty()
    {
        _dirtyCapturedNodes.Clear();
    }
}
