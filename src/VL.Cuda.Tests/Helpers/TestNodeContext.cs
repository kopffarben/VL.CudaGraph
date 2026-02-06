using System.Runtime.CompilerServices;
using VL.Core;

namespace VL.Cuda.Tests.Helpers;

/// <summary>
/// Creates fake NodeContext instances for testing. Uses GetUninitializedObject
/// because NodeContext requires AppHost which is abstract with internal members.
/// All fields are default (null/0) — sufficient for Phase 2 where NodeContext
/// is only stored, never accessed.
/// </summary>
internal static class TestNodeContext
{
    public static NodeContext Create()
        => (NodeContext)RuntimeHelpers.GetUninitializedObject(typeof(NodeContext));
}
