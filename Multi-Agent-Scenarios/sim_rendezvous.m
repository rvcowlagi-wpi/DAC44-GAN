%{
SOFTWARE LICENSE
----------------
Copyright (c) 2023 by 
	Raghvendra V Cowlagi

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
Multi-agent rendezvous simulation based on 
	M. Ji and M. Egerstedt, "Distributed Coordination Control of Multiagent
	Systems While Preserving Connectedness," in IEEE Transactions on Robotics,
	vol. 23, no. 4, pp. 693-703, Aug. 2007, doi: 10.1109/TRO.2007.900638.  
%}

function sim_rendezvous()
close all; clc;

%% Simulation Options and Parameters
WORKSPACE_SIZE		= 1;
N_AGENTS			= 5;
MAX_COMMS_DISTANCE	= 0.01;													% Disk size for distance-based comms topology 
DT_					= 1E-3;

SIM_OPTIONS.measNoise		= false;
SIM_OPTIONS.commsDelay		= false;		
SIM_OPTIONS.commsTopology	= 'distance';									% Options: {'fixed', 'distance'}
SIM_OPTIONS.duration		= 10;


%% Initialization
x_		= -WORKSPACE_SIZE + 2*WORKSPACE_SIZE*rand(2, N_AGENTS);
t_		= 0;
nIter	= 1;
nExpIter= round( SIM_OPTIONS.duration / DT_ );

adjacencyMatrix = zeros(N_AGENTS);
if strcmp(SIM_OPTIONS.commsTopology, 'distance')
	proximity_graph()
end

%----- Store results
tStore	= zeros(nExpIter, 1);
xStore	= zeros(2, N_AGENTS, nExpIter);
amStore	= zeros(N_AGENTS, N_AGENTS, nExpIter);

%% Simulate
% while (1)
	tStore(nIter)		= t_;
	xStore(:, :, nIter)	= x_;
	amStore(:, :, nIter)= adjacencyMatrix;
% 
% 	t_ = t_ + DT_;
% 	if t_ > SIM_OPTIONS.duration
% 		break;
% 	end
% 	
% 	%----- Update topology if distance-based
% 	if strcmp(SIM_OPTIONS.commsTopology, 'distance')
% 		proximity_graph()
% 	end
% 
% 	%----- Update state
% 	for m1 = 1:N_AGENTS
% 
% 		xmDot	= agent_dynamics(t_, x_(:, m1), xFriends);
% 		x_(:, m1)= x_(:, m1) + xmDot*DT_;
% 	end
% 
% 	nIter	= nIter + 1;
% end

%% Plot
figure('Name','Multi-Agent Rendezvous', 'Units','normalized','OuterPosition',[0.1 0.1 0.6 0.8])

%% Agent dynamics

	%% Distance-based comms	network graph
	function proximity_graph()
		for m2 = 1:N_AGENTS
			distance_ = x_ - x_(:, m2);
			distance_ = distance_(1, :).^2 + distance_(2, :).^2;
			isProximal= distance_ <= MAX_COMMS_DISTANCE^2;

			adjacencyMatrix(m2, isProximal)	= 1;
			adjacencyMatrix(m2, m2)			= 0;
		end
		adjacencyMatrix	= adjacencyMatrix + adjacencyMatrix';
	end

end
