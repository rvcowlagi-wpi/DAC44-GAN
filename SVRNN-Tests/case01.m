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

function xSim = case01(n_, nState, nTimePts, tFin)


rng(n_*nState, 'twister');
xSim	= zeros(nState, nTimePts + 1);
x		= -5 + 10*rand(nState, 1);

xSim(:, 1)	= x;

%% Stable linear system with randomly chosen eigenvalues
nComplexPair= floor(nState/4);
nReal		= nState - 2*nComplexPair;

rng(1, 'twister');
realEVs		= -5 + 4*rand(nReal, 1);
complexPart = 5*rand(nComplexPair, 1);
realPart	= -5 + 4*rand(nComplexPair, 1);

%----- Make a temporary A  matrix in block form
A_ = zeros(nState);
for m1 = 1:nReal
	A_(m1, m1) = realEVs(m1);
end
for m1 = 1:nComplexPair
	m2 = nReal + 2*(m1 - 1) + 1;
	p_ = poly([realPart(m1) + complexPart(m1)*1i; realPart(m1) - complexPart(m1)*1i]);
	A_( m2:m2+1, m2:m2+1 ) = [0 1; -p_(3) -p_(2)];
end

%----- Random symmetric positive definite matrix
S	= sprandsym(nState, 1);

%----- Get A matrix from a similarity transformation 
A	= S * A_ / S;

%----- Noise transformation
G	= 0.1*randn(nState, 1);

%----- Time step and noise step
dt_	= tFin / nTimePts;
ndt_= 5;

rng(n_, 'twister');
w	= -1 + 2*rand
for m1 = 2:(nTimePts + 1)
	t	= (m1 - 1)*dt_;
	u	= 0;
	if ~mod(m1, ndt_)
		w	= -1 + 2*rand
	end

	x	= rk4_step(t, x, u, w);
	xSim(:, m1) = x;
end


	function xDot_ = system_(t_, x_, u_, w_)
		xDot_ = A*x_ + G*w_;
	end


	function x_ = rk4_step(t_, x_, u_, w_)

		a1	= 0.5;		a2	= 0.5;		a3	= 1;
		b1	= 0.5;		b2	= 0;		b3	= 0.5;
		b4	= 0;		b5	= 0;		b6	= 1;
		g1	= 1/6;		g2	= 1/3;		g3	= 1/3;		g4	= 1/6;

		k1	= dt_ * system_(t_, x_, u_, w_);
		k2	= dt_ * system_(t_ + a1*dt_, x_ + b1*k1, u_, w_);
		k3	= dt_ * system_(t_ + a2*dt_, x_ + b2*k1 + b3*k2, u_, w_);
		k4	= dt_ * system_(t_ + a3*dt_, x_ + b4*k1 + b5*k2 + b6*k3, u_, w_);

		x_ = x_ + g1*k1 + g2*k2 + g3*k3 + g4*k4;
	end
end