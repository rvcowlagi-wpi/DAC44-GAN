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
This script generates trajectory examples of an uncontrolled dynamical
system.
%}

clear variables; close all; clc;

nTrials		= 1;
nTimePts	= 10;
tFin		= 1;
nState		= 2;
dataSize	= nTimePts*nState;
case_		= 1;

caseName	= ['case_' num2str(case_, '%5.2i')];
caseHandle	= str2func(caseName);

foldername_ = ['Data/' caseName];
if ~exist(foldername_, 'dir')
	mkdir(foldername_)
else
	delete([foldername_ '/*.csv'])
end

for m = 1:nTrials
	xSim	= caseHandle(m, nState, nTimePts, tFin);
	filename_ = [caseName num2str(m) '.csv']

	plot(linspace(0,tFin, nTimePts+1), xSim, 'LineWidth', 2)

	% writematrix(traj_k1, filename_ );
end

return


%% Plot a randomly chosen trajectory
this_trial	= 1 + round(rand*(n_traj_examples - 1));
% filename_	= [foldername_ '/lti_1d_points' num2str(this_trial) '.csv'];
% this_traj	= readmatrix(filename_);

this_traj = traj_k1;

figure('units', 'normalized', 'OuterPosition', [0.05 0.05 0.6 0.6]);
hold on; grid on; axis equal; %axis tight;
plot(t_sim, this_traj(:, 1), 'LineWidth', 2); hold on;
plot(t_sim(1:5:end), this_traj(1:5:end, 1), '.', 'MarkerSize', 40)
xlabel('Time~~$t$','Interpreter','latex');
ylabel('Output~~$y_1$', 'Interpreter', 'latex'); xlim([0 10])
ax = gca;
ax.FontName = 'Arial';
ax.FontSize = 20;
% figure('units', 'normalized', 'OuterPosition', [0.05 0.05 0.6 0.6]);


return

this_trial = unique( 1 + round(rand(50, 1)*(n_traj_examples - 1)) );
for m1 = 1:3
	for m2 = 1:6
		indx = 6*(m1 - 1) + m2;
		filename_	= [foldername_ '/lti_1d_points' num2str(this_trial(indx)) '.csv'];
		this_traj	= readmatrix(filename_);

		subplot(3, 6, indx)
		plot(t_sim, this_traj(:, 1), 'LineWidth', 2)
		xlabel('$t$', 'Interpreter','latex')
		ylabel('$x$', 'Interpreter','latex')
	end
end
% exportgraphics(gcf, 'lti_1d_real_examples.png')
