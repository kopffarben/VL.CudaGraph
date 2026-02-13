using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Main
{
    /// <summary>
    /// Delegate used to create the initial state of the region.
    /// </summary>
    /// <typeparam name="TState">The type of the state object.</typeparam>
    /// <param name="stateOutput">The newly created state.</param>
    public delegate void CreateRegionHandler<TState>(out TState stateOutput);

    /// <summary>
    /// Delegate used to update the region logic based on the state.
    /// </summary>
    /// <typeparam name="TState">The type of the state object.</typeparam>
    /// <typeparam name="TOutput">The type of the output produced.</typeparam>
    /// <param name="stateInput">The current state passed in.</param>
    /// <param name="stateOutput">The updated state passed out.</param>
    /// <param name="output">The calculated output.</param>
    public delegate void UpdateRegionHandler<TState, TOutput>(TState stateInput, TOutput input, out TState stateOutput, out TOutput output);

    /// <summary>
    /// Delegate used to update the region logic based on the state.
    /// </summary>
    /// <typeparam name="TState">The type of the state object.</typeparam>
    /// <typeparam name="TOutput">The type of the output produced.</typeparam>
    /// <param name="stateInput">The current state passed in.</param>
    /// <param name="stateOutput">The updated state passed out.</param>
    /// <param name="output">The calculated output.</param>
    public delegate void SecondMomentRegionHandler<TState, TOutput>(TState stateInput, TOutput input, out TState stateOutput, out TOutput output);
}
