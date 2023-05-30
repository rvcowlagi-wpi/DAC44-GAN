%{
SOFTWARE LICENSE
----------------
Copyright (c) 2023 by 
	Raghvendra V. Cowlagi

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
Large trajectory data generator for simple systems with process noise. This
script generates a .mat file. See " " for writing .csv files from this
dataset.
%}

function datagen_general()
close all; clc;

%----- System
[nState, tFinal, fDynamics, filename_] = system_selector(id_);

%----- Dataset characteristics
nExamples		= 1;
nDiscretization	= 100;
dataSize		= nDiscretization*nState;

%---- Initialize
trajectoryData = zeros(dataSize, nExamples);
for m = 1:nExamples
	xSim	= zeros(nState, nDiscretization);
	my_simulator()

	x0	= -5 + 10*rand(n_state, 1);
	A	= [0 1; -5 -1];

	[~, xSim] = ode45(@(t,x) fDynamics, linspace(0, tFinal, nDiscretization), x0);
	trajectoryData(:, m) = xSim(:);
end
tSim	= linspace(0, tFinal, nDiscretization);

% save(filename_,"tSim","trajectoryData")

% prompt_				= ['\n\n ***** Erase all *.csv files from ' foldername_ '? (y/n) ***** \n'];
% confirmDeleteFolder = input(prompt_, "s");

% if strcmp(confirmDeleteFolder, 'y')
% 	delete([foldername_ '/*.csv'])
% else
% 	foldername_ = ['Data/lti_2d_trajectories_' num2str(round(posixtime(datetime)))];
% end

% for k1 = 1:nExamples
% 	traj_k1		= zeros(nDiscretization, nState);
% 	for k2 = 1:nState
% 		traj_k1(:, k2) = trajectoryData( ...
% 			(1 + (k2 - 1)*nDiscretization):(k2*nDiscretization), k1 );
% 	end
% 	
% 	filename_ = [foldername_ '/example' num2str(k1) '.csv'];
% 
% 	writematrix(traj_k1, filename_ );
% end

	%% Selector
	function [nState, tFinal, fDynamics, filename_] = system_selector(id_)
		switch id_
			case 1
				nState		= 1;
				tFinal		= 10;
				fDynamics	= @my_lti1d;
				filename_	= 'Data/lti1d';
			case 2
				nState		= 2;
				tFinal		= 10;
				fDynamics	= @my_lti2d;
				filename_	= 'Data/lti2d';
			case 3
				nState		= 2;
				tFinal		= 10;
				fDynamics	= @my_vanderpol;
				filename_	= 'Data/vdp';
			case 4
				nState		= 2;
				tFinal		= 10;
				fDynamics	= @my_ipoc;
				filename_	= 'Data/ipoc';
			case 5
				nState		= 2;
				tFinal		= 10;
				fDynamics	= @lti2d;
				filename_	= 'Data/lti2d';
		end
	end
	
	%% Simulator
	function my_simulator()	
		xk			= x0;
		xSim(:, 1)	= xk;
		for k_ = 2:nDiscretization
			xk			= fDynamics(xk);
			xSim(:, k)	= xk;
		end
	end
	
	%% LTI 1D
	function xk1 = my_lti1d(xk)
		xk1 = -a*dt_*xk + randn;
	end
	
	%% LTI 2D
	
	%% LTI 

end