using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Stride.Core.Mathematics;

namespace Main;
public static class Helper
{
    public static Array CreateArray(object value, int capacity)
    {
        return value switch
        {
            null => throw new ArgumentNullException(nameof(value)),
            // Primitives
            bool x => new bool[capacity],
            char x => new char[capacity],
            byte x => new byte[capacity],
            sbyte x => new sbyte[capacity],
            short x => new short[capacity],
            ushort x => new ushort[capacity],
            int x => new int[capacity],
            uint x => new uint[capacity],
            long x => new long[capacity],
            ulong x => new ulong[capacity],
            float x => new float[capacity],
            double x => new double[capacity],
            // Stride Mathematics
            Vector2 x => new Vector2[capacity],
            Vector3 x => new Vector3[capacity],
            Vector4 x => new Vector4[capacity],
            Int2 x => new Int2[capacity],
            Int3 x => new Int3[capacity],
            Int4 x => new Int4[capacity],
            Quaternion x => new Quaternion[capacity],
            Matrix x => new Matrix[capacity],
            Color x => new Color[capacity],
            Color4 x => new Color4[capacity],
            Rectangle x => new Rectangle[capacity],
            RectangleF x => new RectangleF[capacity],
            _ => throw new ArgumentException($"Unsupported type: {value.GetType().FullName}", nameof(value))
        };
    }

    public static ISpread ToSpread(this Array array)
    {
        return array switch
        {
            bool[] x => Spread.Create(x),
            char[] x => Spread.Create(x),
            byte[] x => Spread.Create(x),
            sbyte[] x => Spread.Create(x),
            short[] x => Spread.Create(x),
            ushort[] x => Spread.Create(x),
            int[] x => Spread.Create(x),
            uint[] x => Spread.Create(x),
            long[] x => Spread.Create(x),
            ulong[] x => Spread.Create(x),
            float[] x => Spread.Create(x),
            double[] x => Spread.Create(x),
            Vector2[] x => Spread.Create(x),
            Vector3[] x => Spread.Create(x),
            Vector4[] x => Spread.Create(x),
            Int2[] x => Spread.Create(x),
            Int3[] x => Spread.Create(x),
            Int4[] x => Spread.Create(x),
            Quaternion[] x => Spread.Create(x),
            Matrix[] x => Spread.Create(x),
            Color[] x => Spread.Create(x),
            Color4[] x => Spread.Create(x),
            Rectangle[] x => Spread.Create(x),
            RectangleF[] x => Spread.Create(x),
             _ => throw new ArgumentException($"Unsupported array type: {array.GetType().FullName}", nameof(array))
        };
    }

    public static ISpreadBuilder CreateSpreadBuilder(object value, int capacity)
    {
        var tmp = Spread.Create(new float[capacity]);

        return value switch
        {
            null => throw new ArgumentNullException(nameof(value)),

            // Primitives
            bool x => new SpreadBuilder<bool>(capacity) { x },
            char x => new SpreadBuilder<char>(capacity) { x },
            byte x => new SpreadBuilder<byte>(capacity) { x },
            sbyte x => new SpreadBuilder<sbyte>(capacity) { x },
            short x => new SpreadBuilder<short>(capacity) { x },
            ushort x => new SpreadBuilder<ushort>(capacity) { x },
            int x => new SpreadBuilder<int>(capacity) { x },
            uint x => new SpreadBuilder<uint>(capacity) { x },
            long x => new SpreadBuilder<long>(capacity) { x },
            ulong x => new SpreadBuilder<ulong>(capacity) { x },
            float x => new SpreadBuilder<float>(capacity) { x },
            double x => new SpreadBuilder<double>(capacity) { x },

            // Stride Mathematics
            Vector2 x => new SpreadBuilder<Vector2>(capacity) { x },
            Vector3 x => new SpreadBuilder<Vector3>(capacity) { x },
            Vector4 x => new SpreadBuilder<Vector4>(capacity) { x },
            Int2 x => new SpreadBuilder<Int2>(capacity) { x },
            Int3 x => new SpreadBuilder<Int3>(capacity) { x },
            Int4 x => new SpreadBuilder<Int4>(capacity) { x },
            Quaternion x => new SpreadBuilder<Quaternion>(capacity) { x },
            Matrix x => new SpreadBuilder<Matrix>(capacity) { x },
            Color x => new SpreadBuilder<Color>(capacity) { x },
            Color4 x => new SpreadBuilder<Color4>(capacity) { x },
            Rectangle x => new SpreadBuilder<Rectangle>(capacity) { x },
            RectangleF x => new SpreadBuilder<RectangleF>(capacity) { x },

            _ => throw new ArgumentException($"Unsupported type: {value.GetType().FullName}", nameof(value))
        };
    }
}
