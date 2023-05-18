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

function rendezvous_plain()
close all; clc;

%% Simulation Options and Parameters
WORKSPACE_SIZE		= 1;
N_AGENTS			= 20;
MAX_COMMS_DISTANCE	= 0.5;													% Disk size for distance-based comms topology 
DT_					= 1E-3;

SIM_OPTIONS.measNoise		= false;
SIM_OPTIONS.commsDelay		= false;		
SIM_OPTIONS.commsTopology	= 'distance';									% Options: {'fixed', 'distance'}
SIM_OPTIONS.duration		= 3;
SIM_OPTIONS.dualScreen		= false;

%% Control Parameters
gainK = 1;

%% Initialization
%----- Make sure initial agent locations are "spread out" yet connected
x_		= zeros(2, N_AGENTS);
x_(:, 1)= -WORKSPACE_SIZE + 2*WORKSPACE_SIZE*rand(2, 1);
m1 = 2;
while m1 <= N_AGENTS
	x_(:, m1)	= -WORKSPACE_SIZE + 2*WORKSPACE_SIZE*rand(2, 1);
	dx	= x_(:, m1) - x_(:, 1:m1-1);
	dx	= (dx(1, :).^2 + dx(2, :) .^2).^(0.5);
	if ((m1 <= 4) && any(dx < MAX_COMMS_DISTANCE)) || ...
			((m1 > 4) && any(dx < MAX_COMMS_DISTANCE) && ...
			(sum(dx > MAX_COMMS_DISTANCE) > m1 - 3))
		m1 = m1 + 1;
	end
end

t_		= 0;
nIter	= 1;
nExpIter= round( SIM_OPTIONS.duration / DT_ );

adjacency_	= zeros(N_AGENTS);
friendship_	= zeros(N_AGENTS);
degree_		= zeros(N_AGENTS);
Laplacian_	= zeros(N_AGENTS);
if strcmp(SIM_OPTIONS.commsTopology, 'distance')
	proximity_graph()
end

%----- Store results
tStore	= zeros(nExpIter, 1);
xStore	= zeros(2, N_AGENTS, nExpIter);
amStore	= zeros(N_AGENTS, N_AGENTS, nExpIter);

%% Simulate
while (1)
	tStore(nIter)		= t_;
	xStore(:, :, nIter)	= x_;
	amStore(:, :, nIter)= adjacency_;

	t_	= t_ + DT_;
	y_	= x_;
	if t_ > SIM_OPTIONS.duration + 0.5*DT_
		break;
	end
	
	%----- Update topology if distance-based
	if strcmp(SIM_OPTIONS.commsTopology, 'distance')
		proximity_graph()
	end

	%----- Update state
	for m1 = 1:N_AGENTS
		xmDot		= agent_dynamics(m1);
		x_(:, m1)	= x_(:, m1) + xmDot*DT_;
	end

	nIter	= nIter + 1;
end

%% Plot
if SIM_OPTIONS.dualScreen
	dispXOffset = 1;
else
	dispXOffset = 0;
end
figure('Name','Multi-Agent Rendezvous', 'Units','normalized', ...
		'OuterPosition', [dispXOffset + 0.1 0.1 0.5 0.8])
ax = gca;
thisColor	= ax.ColorOrder(1,:);
myFont		= 'Times New Roman';

grHdlTmp = plot(0,0); hold on; grid on; axis equal
xlim(1.5*WORKSPACE_SIZE*[-1 1]); ylim(1.5*WORKSPACE_SIZE*[-1 1])
ax.FontName = myFont;
ax.FontSize = 16;
delete(grHdlTmp)

nowText			= num2str(round(posixtime(datetime)));
videoFileName	= ['Results/202305/rendezvous_plain_run' nowText '.mp4'];
dataFileName	= ['Results/202305/rendezvous_plain_run' nowText '.mat'];

vid_	= VideoWriter(videoFileName);
vid_.open();
nSkip	= round(nIter / 500);
for n1 = 1:nSkip:nIter
	cla();
	plot_network(n1);
	plot_nodes(n1);

	text(-1.42*WORKSPACE_SIZE, 1.42*WORKSPACE_SIZE, ['$t = $' num2str(tStore(n1))], ...
		"FontName", myFont, 'FontSize', 16, 'Interpreter', 'latex')
	drawnow();

	vid_.writeVideo(getframe(gcf));
end
vid_.close();

save(dataFileName)

	%% Distance-based comms	network graph
	function proximity_graph()
		adjacency_	= zeros(N_AGENTS);
		for m2 = 1:N_AGENTS
			distance_ = x_ - x_(:, m2);
			distance_ = ( distance_(1, :).^2 + distance_(2, :).^2 ).^(0.5);
			isProximal= distance_ <= MAX_COMMS_DISTANCE;

			adjacency_(m2, isProximal)	= 1;
			adjacency_(m2, m2)			= 0;
		end
		degree_		= diag(sum(adjacency_, 1));
		Laplacian_	= degree_ - adjacency_;
		friendship_ = adjacency_;
	end

	%% Draw network topology
	function plot_network(ell_)
		for m2 = 1:N_AGENTS
			for m3 = (m2 + 1) : N_AGENTS
				if ~amStore(m2, m3, ell_), continue; end
				line([xStore(1, m2, ell_), xStore(1, m3, ell_)], ...
					[xStore(2, m2, ell_), xStore(2, m3, ell_)]);
			end
		end
	end

	%% Draw nodes
	function plot_nodes(ell_)
		plot(xStore(1, :, ell_), xStore(2, :, ell_), ...
			'Marker','o', 'MarkerFaceColor', thisColor, ...
			'MarkerEdgeColor', thisColor, 'MarkerSize', 25, ...
			'LineStyle','none');

		for m2 = 1:N_AGENTS
			textXOffset = 0.01;
			if length(num2str(m2)) > 1, textXOffset = textXOffset + 0.02; end
			text(xStore(1, m2, ell_) - textXOffset, xStore(2, m2, ell_), ...
				num2str(m2), ...
				'FontSize', 13, 'FontName', 'Times New Roman', ...
				'FontWeight', 'bold', 'Color', 'w')
		end
	end

	%% Agent dynamics
	function xmDot = agent_dynamics(m_)
		%-- Find friends
		tmp_		= 1:N_AGENTS;
		friendAgents= tmp_(friendship_(m_, :) > 0);

		%-- Update based on friend data
		um_ = 0;
		for m2 = friendAgents
			um_ = um_ - gainK*(y_(:, m_) - y_(:, m2));
		end
		xmDot = um_;
	end

end
