using System;
using System.Collections.Generic;

namespace VL.Cuda.Core.Blocks;

/// <summary>
/// Typed scalar parameter with change tracking. Value changes fire the
/// ValueChanged event so the BlockBuilder/CudaContext can mark parameters dirty.
/// </summary>
public sealed class BlockParameter<T> : IBlockParameter where T : unmanaged
{
    private T _value;

    public string Name { get; }
    public Type ValueType => typeof(T);

    /// <summary>
    /// Fired when the value changes. Subscribers: BlockBuilder wiring → CudaContext.OnParameterChanged.
    /// </summary>
    public event Action<BlockParameter<T>>? ValueChanged;

    public T TypedValue
    {
        get => _value;
        set
        {
            if (!EqualityComparer<T>.Default.Equals(_value, value))
            {
                _value = value;
                IsDirty = true;
                ValueChanged?.Invoke(this);
            }
        }
    }

    object IBlockParameter.Value
    {
        get => _value;
        set => TypedValue = (T)value;
    }

    public bool IsDirty { get; private set; }

    public void ClearDirty() => IsDirty = false;

    /// <summary>
    /// The kernel node ID this parameter maps to (set by BlockBuilder.Commit).
    /// </summary>
    internal Guid KernelNodeId { get; set; }

    /// <summary>
    /// The kernel parameter index this maps to (set by BlockBuilder.Commit).
    /// </summary>
    internal int KernelParamIndex { get; set; }

    public BlockParameter(string name, T defaultValue = default)
    {
        Name = name;
        _value = defaultValue;
    }

    public override string ToString() => $"{Name}: {_value}";
}
