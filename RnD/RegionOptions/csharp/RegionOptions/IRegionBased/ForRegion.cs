using System.Linq;
using System.Reflection;
using System.Text.Json;
using System.Text.RegularExpressions;
using VL.AppServices.CompilerServices;
using VL.AppServices.CompilerServices.CustomRegion;
using VL.AppServices.Hotswap;
using VL.Core.PublicAPI;

namespace Main;

[ProcessNode( FragmentSelection = FragmentSelection.Explicit)]
[Region(SupportedBorderControlPoints = /*ControlPointType.Accumulator |*/ ControlPointType.Splicer , TypeConstraint = "Spread", TypeConstraintIsBaseType = true)]
public class ForRegion : IRegion<ForRegionInterface>, IDisposable
{
    private NodeContext _nodeContext;
    private ForRegionInterface? _patchInlay;
    private Func<ForRegionInterface>? _patchInlayFactory;
    private readonly Dictionary<InputDescription,  IList<object>> _inputSplicers  = new();
    private readonly Dictionary<OutputDescription, IList<object>> _outputSplicers = new();
    private readonly Dictionary<InputDescription,  object>              _inputAccumulator  = new();
    private readonly Dictionary<OutputDescription, object> _outputAccumulator = new();
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<Type, Func<IEnumerable<object>, object>> _spreadCastCache = new();
    private int _currentIndex = 0;

    [Fragment(Order = 0)]
    public ForRegion( [Pin(Visibility = VL.Model.PinVisibility.Hidden)] NodeContext nodeContext )
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
            var count = _inputSplicers.Select(kv => kv.Value.Count()).Min();
            for (_currentIndex = 0; _currentIndex < count; _currentIndex++)
            {
                _patchInlay.Update(_currentIndex, out var _break);
                if (_break)
                {
                    break;
                }
            }
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
        if (description.IsSplicer && outerValue is ISpread spread)
        {
            _inputSplicers[description] = spread.Cast<object>().ToList();
        }
        else
        {
            _inputAccumulator[description] = outerValue!;
        }
    }

    /// <inheritdoc />
    public void RetrieveInput(in InputDescription description, ForRegionInterface patchInlay, out object? innerValue)
    {
        if (description.IsSplicer)
        {
            innerValue = _inputSplicers[description][_currentIndex];
        }
        else
        {
            innerValue = _inputAccumulator[description];
        }
    }

    /// <inheritdoc />
    public void AcknowledgeOutput(in OutputDescription description, ForRegionInterface patchInlay, object? innerValue)
    {
       if (description.IsSplicer)
        {
            if (!_outputSplicers.ContainsKey(description))
            {
                _outputSplicers[description] = new List<object>();
            }
            else if (_outputSplicers[description].Count <= _currentIndex)
            {
                _outputSplicers[description].Add(innerValue);
            }
            else
            {
                _outputSplicers[description][_currentIndex] = innerValue;
            }
        }
        else
        {
            _outputAccumulator[description] = innerValue!;
        }
    }

    /// <inheritdoc />
    public void RetrieveOutput(in OutputDescription description, out object? outerValue)
    {
        if (description.IsSplicer)
        {
            var list = _outputSplicers[description];
            if (list.Count() >  _currentIndex + 1)
            {
                int maxCount = _currentIndex + 1;
                ((List<object>)list).RemoveRange(maxCount, list.Count() - (maxCount));
            }
            outerValue = _spreadCastCache.CastSpread(list, description.InnerType);
        }
        else
        {
            outerValue = null;
        }

    }

    /// <summary>
    /// Sets the factory function used to create the patch inlay instance.
    /// </summary>
    /// <param name="patchInlayFactory">The factory function.</param>
    public void SetPatchInlayFactory(Func<ForRegionInterface> patchInlayFactory)
    {
        this._patchInlayFactory = patchInlayFactory;
    }
    #endregion IRegion<TestReginInterface> Members
   
}

public static class Helper
{
    public static object CastSpread(this System.Collections.Concurrent.ConcurrentDictionary<Type, Func<IEnumerable<object>, object>> _spreadCastCache, IEnumerable<object> source, Type destType)
    {
        var converter = _spreadCastCache.GetOrAdd(destType, t =>
        {
            // Parameter: source (IEnumerable<object>)
            var sourceParam = System.Linq.Expressions.Expression.Parameter(typeof(IEnumerable<object>), "source");

            // Method: Enumerable.Cast<destType>(source)
            var castMethod = typeof(Enumerable)
                .GetMethod(nameof(Enumerable.Cast))!
                .MakeGenericMethod(t);

            // Call: Cast<destType>(source)
            var castCall = System.Linq.Expressions.Expression.Call(castMethod, sourceParam);

            // Method: Spread.ToSpread<destType>(IEnumerable<destType>)
            // Reusing your specific lookup logic to avoid AmbiguousMatchException
            var toSpreadMethod = typeof(Spread).GetMethods()
                .First(m => m.Name == "ToSpread"
                            && m.GetParameters().Length == 1
                            && m.GetParameters()[0].ParameterType.Name.Contains("IEnumerable"))
                .MakeGenericMethod(t);

            // Call: ToSpread<destType>(castResult)
            var toSpreadCall = System.Linq.Expressions.Expression.Call(toSpreadMethod, castCall);

            // Return as object
            var convertToObject = System.Linq.Expressions.Expression.Convert(toSpreadCall, typeof(object));

            // Compile to delegate
            return System.Linq.Expressions.Expression.Lambda<Func<IEnumerable<object>, object>>(
                convertToObject, sourceParam).Compile();
        });

        return converter(source);
    }
}