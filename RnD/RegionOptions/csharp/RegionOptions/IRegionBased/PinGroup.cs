using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using VL.Lang.PublicAPI;
using VL.Model;

namespace Main;

[ProcessNode()]
public class PinGroupTest
{
    private readonly NodeContext _nodeContext;
    private readonly UniqueId _nodeId;

    public PinGroupTest([Pin(Visibility = VL.Model.PinVisibility.Hidden)] NodeContext nodeContext)
    {
        _nodeContext = nodeContext; 
        _nodeId = nodeContext.Path.Stack.Peek();

        var pingroupBuilder = SessionNodes.CurrentSolution.ModifyPinGroup(_nodeId, "MyInputPinGroup", true);
        pingroupBuilder.Add("Floaty", "Float");
        pingroupBuilder.Add("Spread", "Spread<Float>");
        pingroupBuilder.Commit().Confirm(SolutionUpdateKind.TweakLast);

    }

    public void Update([Pin(Name = "MyInputPinGroup", PinGroupKind = PinGroupKind.Dictionary)] ImmutableDictionary<string, object> input)
    {
        SessionNodes.CurrentSolution
            .SetPinValue(_nodeId, "Floaty", 3.14f)
            .SetPinValue(_nodeId, "Spread", Spread.Create(new float[] { 1f, 2f, 3f }))
            .Confirm(SolutionUpdateKind.DontCompile);
    }

}
