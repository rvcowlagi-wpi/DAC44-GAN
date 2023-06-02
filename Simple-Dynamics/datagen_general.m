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
script generates a .mat file. See "datawrite_general.m" for writing .csv
files from this dataset.
%}

function datagen_general()
close all; clc;

%----- System
id_			= 1;

nState		= [];
tFinal		= [];
fDynamics	= [];
filename_	= [];
syspar_		= [];
x0			= [];
system_selector(id_);

%----- Dataset characteristics
nExamples		= 10;
nDiscretization	= 100;														% Can be sampled down when writing to csv
dataSize		= nDiscretization*nState;
dt_				= tFinal / nDiscretization;									% Time step

%---- Initialize
trajectoryData = zeros(dataSize, nExamples);
for m = 1:nExamples
	xSim	= zeros(nDiscretization, nState);

	parameter_selector(id_)
	my_simulator()

	trajectoryData(:, m) = xSim(:);
end
tSim	= linspace(0, tFinal, nDiscretization);


%% Plot a few trajectories
nTrajToPlot = nExamples;

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

	%% System selector
	function system_selector(id_)
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

	%% Parameter selector
	function parameter_selector(id_)
		switch id_
			case 1
				x0			= -5 + 10*randn;
				syspar_.a	= -5;
				syspar_.q	= 0.01;
			case 2
				syspar_.A	= [0 1; -5 -1];
			case 3
				syspar_		= [];
			case 4
				syspar_		= [];
			case 5
				syspar_		= [];
		end
	end
	
	%% Simulator
	function my_simulator()	
		xk			= x0;
		xSim(:, 1)	= xk;
		for k_ = 2:nDiscretization
			xk			= fDynamics(xk);
			xSim(k_, :)	= xk';
		end
	end
	
	%% LTI 1D
	function xk1 = my_lti1d(xk)
		xk1 = -syspar_.a*dt_*xk + sqrt(syspar_.q)*randn;
	end
	
	%% LTI 2D
	
	%% LTI 

end