%{
SOFTWARE LICENSE
----------------
Copyright (c) 2023 by Raghvendra V. Cowlagi

Permission is hereby granted to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in
the Software, including the rights to use, copy, modify, merge, copies of
the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:  

* The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
* The Software, and its copies or modifications, may not be distributed,
published, or sold for profit. 
* The Software, and any substantial portion thereof, may not be copied or
modified for commercial or for-profit use.

The software is provided "as is", without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose and noninfringement. In no event shall the
authors or copyright holders be liable for any claim, damages or other
liability, whether in an action of contract, tort or otherwise, arising
from, out of or in connection with the software or the use or other
dealings in the software.      


PROGRAM DESCRIPTION
-------------------
System				: Linear time invariant
Process noise		: Yes, uniformly distributed
Measurement noise	: No
Unmodeled dynamics	: No
%}

function xSim_ = case_01(nTrial_, nState, nTimePts)

x0_ = -5 + 10*rand(nState, 1);

ode45(@(t, x) system_(t, x, A), linspace(0, 10, nTimePts), x0_);

function xDot_ = system_(t__, x__)