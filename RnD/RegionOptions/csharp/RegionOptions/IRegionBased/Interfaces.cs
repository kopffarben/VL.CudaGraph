using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Main;

public interface LFOInterface
{
    void Update(float Period, bool Pause, bool Reset, out float Result);
}

public interface ForRegionInterface
{
    void Update(int Index, out bool Break);
}