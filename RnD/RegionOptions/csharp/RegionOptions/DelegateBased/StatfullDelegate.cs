// For examples, see:
// https://thegraybook.vvvv.org/reference/extending/writing-nodes.html#examples

namespace Main;

/// <summary>
/// Defines a region function by holding references to create and update delegates.
/// </summary>
/// <typeparam name="TState">The type of the state used by the region.</typeparam>
/// <typeparam name="TOutput">The type of the input/output data.</typeparam>
[ProcessNode(HasStateOutput = true)]
public class RegionFunction<TState, TOutput> 
{
    /// <summary>
    /// The delegate used to create the initial state.
    /// </summary>
    public CreateRegionHandler<TState> create;

    /// <summary>
    /// The delegate used to update the state and compute output.
    /// </summary>
    public UpdateRegionHandler<TState, TOutput> update;

    /// <summary>
    /// Configures the region function with the specified delegates.
    /// </summary>
    /// <param name="create">The delegate to create the state if it doesn't exist or upon reset.</param>
    /// <param name="update">The delegate to execute the update logic.</param> 
    public void Update(CreateRegionHandler<TState> create, UpdateRegionHandler<TState, TOutput> update)
    {
       this.create = create;
       this.update = update;
    }
}

/// <summary>
/// Manages the execution and lifecycle of a stateful region function defined by delegates.
/// </summary>
/// <typeparam name="TState">The type of the internal state.</typeparam>
/// <typeparam name="TOutput">The type of the input/output data.</typeparam>
[ProcessNode]
public class RegionFunctionInvoke<TState, TOutput> : IDisposable
{
    TState FState;
    TOutput FLastOutput;

    /// <summary>
    /// Executes the provided region function, managing its state and producing output.
    /// </summary>
    /// <param name="function">The region function definition to invoke.</param>
    /// <param name="input">The input value to pass to the update delegate.</param>
    /// <param name="output">The resulting output from the current logic or the last known valid output.</param>
    /// <param name="enabled">If true, the update logic is executed. If false, the state remains unchanged and last output is returned.</param>
    /// <param name="reset">If true, the current state is disposed and recreated using the create delegate.</param>
    public void Update(RegionFunction<TState, TOutput> function, TOutput input, out TOutput output, bool enabled = true, bool reset = false)
    {
        if (FState == null || reset)
        {
            Dispose();
            function.create(out this.FState);
        }
        if (enabled && FState != null)
        {
            function.update(FState, input, out this.FState, out this.FLastOutput);
        }

        output = this.FLastOutput;
    }

    /// <summary>
    /// Disposes of the resources managed by this instance.
    /// </summary>
    public void Dispose()
    {
        DisposeInternalState();
    }

    /// <summary>
    /// Disposes the underlying state object if it implements IDisposable.
    /// </summary>
    private void DisposeInternalState()
    {
        try
        {
            (FState as IDisposable)?.Dispose();
        }
        finally
        {
            FState = default(TState);
        }
    }
}