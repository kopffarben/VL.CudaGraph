using System.Collections.Generic;

namespace VL.Cuda.Core.Context.Services;

/// <summary>
/// Tracks dirty state for the CUDA pipeline. Subscribes to StructureChanged
/// events from BlockRegistry and ConnectionGraph. CudaEngine reads these
/// flags each frame to decide: Cold Rebuild vs Hot/Warm Update vs no-op.
/// </summary>
public sealed class DirtyTracker
{
    private readonly HashSet<DirtyParameter> _dirtyParameters = new();

    public bool IsStructureDirty { get; private set; } = true; // Start dirty → first build
    public bool AreParametersDirty => _dirtyParameters.Count > 0;

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
    /// Get all dirty parameters and clear.
    /// </summary>
    public IReadOnlySet<DirtyParameter> GetDirtyParameters()
    {
        return _dirtyParameters;
    }

    /// <summary>
    /// Clear structure dirty flag after Cold Rebuild.
    /// Also clears parameter dirty since rebuild applies all current values.
    /// </summary>
    public void ClearStructureDirty()
    {
        IsStructureDirty = false;
        _dirtyParameters.Clear();
    }

    /// <summary>
    /// Clear parameter dirty flags after Hot/Warm Update.
    /// </summary>
    public void ClearParametersDirty()
    {
        _dirtyParameters.Clear();
    }
}
