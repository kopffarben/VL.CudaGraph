using System.Collections.Generic;

namespace VL.Cuda.Core.Context.Services;

/// <summary>
/// Tracks dirty state for the CUDA pipeline. Subscribes to StructureChanged
/// events from BlockRegistry and ConnectionGraph. CudaEngine reads these
/// flags each frame to decide: Cold Rebuild vs Code Rebuild vs Recapture vs Hot/Warm Update vs no-op.
/// Priority: Structure > Code > Recapture > Parameters.
/// </summary>
public sealed class DirtyTracker
{
    private readonly HashSet<DirtyParameter> _dirtyParameters = new();
    private readonly HashSet<DirtyCapturedNode> _dirtyCapturedNodes = new();
    private readonly HashSet<DirtyCodeEntry> _dirtyCodeEntries = new();

    public bool IsStructureDirty { get; private set; } = true; // Start dirty -> first build
    public bool IsCodeDirty => _dirtyCodeEntries.Count > 0;
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
    /// Mark a kernel source as dirty (code changed, needs recompilation).
    /// </summary>
    public void MarkCodeDirty(DirtyCodeEntry entry)
    {
        _dirtyCodeEntries.Add(entry);
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
    /// Get all dirty code entries (kernels needing recompilation).
    /// </summary>
    public IReadOnlySet<DirtyCodeEntry> GetDirtyCodeEntries()
    {
        return _dirtyCodeEntries;
    }

    /// <summary>
    /// Clear structure dirty flag after Cold Rebuild.
    /// Also clears all other dirty flags since rebuild applies all current values.
    /// </summary>
    public void ClearStructureDirty()
    {
        IsStructureDirty = false;
        _dirtyParameters.Clear();
        _dirtyCapturedNodes.Clear();
        _dirtyCodeEntries.Clear();
    }

    /// <summary>
    /// Clear code dirty flags after Code Rebuild.
    /// Code rebuild triggers a Cold Rebuild, so also clears parameters and recapture.
    /// </summary>
    public void ClearCodeDirty()
    {
        _dirtyCodeEntries.Clear();
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
