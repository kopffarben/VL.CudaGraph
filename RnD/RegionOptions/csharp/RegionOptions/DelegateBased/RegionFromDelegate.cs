// For examples, see:
// https://thegraybook.vvvv.org/reference/extending/writing-nodes.html#examples

namespace Main;


/// <summary>
/// A process node that manages a stateful region via creation and update delegates.
/// </summary>
/// <typeparam name="TState">The type of the internal state.</typeparam>
/// <typeparam name="TOutput">The type of the output value.</typeparam>
[ProcessNode]
public class RegionFromDelegate<TState, TOutput> : IDisposable
{
    TState FState;
    TOutput FLastOutput;

    /// <summary>
    /// Updates the region state and produces an output.
    /// </summary>
    /// <param name="create">The delegate to create the state if it doesn't exist or upon reset.</param>
    /// <param name="update">The delegate to execute the update logic.</param>
    /// <param name="FLastOutput">The resulting output from the current or last valid update.</param>
    /// <param name="enabled">If true, the update delegate is executed.</param>
    /// <param name="reset">If true, the internal state is disposed and recreated.</param>
    public void Update(CreateRegionHandler<TState> create, UpdateRegionHandler<TState, TOutput> update, TOutput input, out TOutput output,   bool enabled = true, bool reset = false)
    {
        if (FState == null || reset)
        {
            Dispose();
            create(out this.FState);
        } 
        if (enabled && FState != null)
        {
            update(FState, input, out this.FState, out this.FLastOutput);
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