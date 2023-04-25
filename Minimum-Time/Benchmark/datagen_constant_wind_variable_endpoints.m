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
This script repeats the minimum time base simulator. Wind is constant, and
endpoints are randomly varied at each trial.
%}

clear variables; close all; clc;

% n_trials = 9;
% parfor m = 1:n_trials
% 	mintime_navigation_base()
% end

% %------ For traditional Zermelo solutions
% n_trials	= 1000;
% size_data	= 407;
% baseline_data = zeros(size_data, n_trials);
% parfor m = 1:n_trials
% 	[sim_result_, solution_found_] = mintime_navigation_datagenerator();
% 	if solution_found_
% 		baseline_data(:, m) = sim_result_;
% 	end
% end


%------ For traditional v/s RSAI comparison
% Constant wind, linear in x2
addpath('GPML')
startup

n_trials	= 1;
size_data	= 6;
comparative_data = zeros(size_data, n_trials);
NS			= 10;
parfor m = 1:n_trials
	fprintf('----- Trial %i -----\n', m)
	sim_result_ = rsai01_comparison_datagen(NS);
	comparative_data(:, m) = sim_result_;
end
filename_ = ['../Results/rsai01_comparison_NS' num2str(NS) '_' ...
	num2str(posixtime(datetime(datestr(now)))) '.mat'];
save(filename_, 'comparative_data')