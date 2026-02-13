// For examples, see:
// https://thegraybook.vvvv.org/reference/extending/writing-nodes.html#examples

namespace Main;

[ProcessNode]
public class MultiMomentRegionFromDelegate<TState, TOutput> : IDisposable
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
    public void Update(CreateRegionHandler<TState> create, UpdateRegionHandler<TState, TOutput> update, SecondMomentRegionHandler<TState, TOutput> second, TOutput input, out TOutput output, bool enabled = true, bool reset = false)
    {
        if (FState == null || reset)
        {
            Dispose();
            create(out this.FState);
        }
        if (enabled && FState != null)
        {
            update(FState, input, out this.FState, out this.FLastOutput);
            // just a chain, for example to calculate a second moment based on the first moment
            second(FState, this.FLastOutput, out this.FState, out this.FLastOutput);
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