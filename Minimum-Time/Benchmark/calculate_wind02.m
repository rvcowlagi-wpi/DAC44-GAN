%{
SOFTWARE LICENSE
----------------
Copyright (c) 2021 by Raghvendra V. Cowlagi

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
The wind velocity considered here is linearly dependent on the x2 position
coordinate of an intermediate axes system that is rotated relative
to the global axes system. 
%}

function [wind_x1, wind_x2, ...
	wind_gradient_11, wind_gradient_12, ...
	wind_gradient_21, wind_gradient_22] = ...
	calculate_wind02(x1_, x2_, params)


% params.const_1 = rotation of the wind
% params.const_2 = rate of change of wind along intermediate y

x2b		= -sin(params.const_1)*x1_ + cos(params.const_1)*x2_;
wind_x1	= params.const_2*x2b*cos(params.const_1);
wind_x2	= params.const_2*x2b*sin(params.const_1);

wind_gradient_11	= ...
	-params.const_2 * cos(params.const_1) * sin(params.const_1) * ...
	ones(size(x1_,1), size(x2_,2));
wind_gradient_12	= ...
	params.const_2 * cos(params.const_1) * cos(params.const_1) * ...
	ones(size(x1_,1), size(x2_,2));
wind_gradient_21	= ...
	-params.const_2 * sin(params.const_1) * sin(params.const_1) * ...
	ones(size(x1_,1), size(x2_,2));
wind_gradient_22	= ...
	params.const_2 * cos(params.const_1) * sin(params.const_1) * ...
	ones(size(x1_,1), size(x2_,2));

end