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
This program implements the minimum-time solution for the Zermelo
navigation problem. The wind calculator function is an input argument.

Required wind calculator template:
[wind_x1, wind_x2, wind_gradient_11, wind_gradient_12, ...
	wind_gradient_21, wind_gradient_22] = wind_fcn(x1_, x2_, params)
%}

function [sim_result_, solution_found_] = ...
	zermelo(x_init, x_term, wind_fcn, wind_fcn_params, solver_params)

hamiltonian_tol = 1E-7;
ode_rel_tol		= 1E-9;
ode_abs_tol		= 1E-6;
bd_fcn_tol		= 1E-6;
just_sim		= false;
n_trials		= 25;
check_h			= true;
if numel(solver_params)
	hamiltonian_tol = solver_params.hamiltonian_tol;
	ode_rel_tol		= solver_params.ode_rel_tol;
	ode_abs_tol		= solver_params.ode_abs_tol;
	bd_fcn_tol		= solver_params.bd_fcn_tol;
	just_sim		= solver_params.just_sim;
	n_trials		= solver_params.n_trials;
	check_h			= solver_params.check_hamiltonian;
end

%% Problem data

%----- Size of the workspace (normalized to [-1, 1] interval)
%----- Aircraft speed (normalized units)
spd_V		= 0.05;

%% Find optimal initial heading and traversal time
%{
	There may be multiple solutions, each correspoding to a local minimum.
	We find as many as possible through randomized initial guesses.
%}
ode_solver_options	= odeset('RelTol', ode_rel_tol, ...
	'AbsTol', ode_abs_tol, 'Events', @wksp_bd_cross_event);
bc_solver_options	= optimoptions('fsolve', ...
	'Display', 'none', 'FunctionTolerance', bd_fcn_tol);

min_cost			= Inf;
y_star				= [Inf; Inf];
min_bd_tol			= Inf;
solution_found_		= 0;
optimal_solutions	= [];

n_ = 0;
% for n_ = 0:n_trials
while (~just_sim)
	clc;

% 	fprintf('Trial: \t %i\n', n_);
% 	fprintf('Number of solutions found: \t %i\n', size(optimal_solutions, 2));
% 	fprintf('Solutions: \n');
% 	disp(optimal_solutions)

	
	%----- Initial guesses for optimal initial heading and traversal time tf
	initial_guess_opt_psi0	= 2*pi*rand;
% 	initial_guess_opt_tf= (1 + 4*rand)*(norm(x_term - x_init) / spd_V);
	initial_guess_opt_tf= (0.75 + 2.5*rand)*(norm(x_term - x_init) / spd_V);

	%----- Wind at initial position
	[wind_x1_init, wind_x2_init, ~, ~, ~, ~] = ...
		wind_fcn(x_init(1), x_init(2), wind_fcn_params);

	%----- Solve for optimal initial heading and traversal time
	[y_opt, fval_, exit_flag] = ...
		fsolve(@boundary_conditions, ...
		[initial_guess_opt_psi0; initial_guess_opt_tf], bc_solver_options);
	
	n_	= n_ + 1;
	if ( (n_ > n_trials) && (size(optimal_solutions, 2) >= 1) ) || ...
			(n_ > 4*n_trials)
		break;
	end

	if ~any(exit_flag == [1 2 3 4])
		%---- Boundary conditions not satisfied
		% This is not a valid solution
		continue;
	end
	
	%----- Validate solution: Check if Hamiltonian remains zero always
	psi0_opt= y_opt(1);
	tf_opt	= y_opt(2);
	q_init	= [x_init; psi0_opt; ...
		-cos(psi0_opt)/(spd_V + wind_x1_init*cos(psi0_opt) + wind_x2_init*sin(psi0_opt)); ...
		-sin(psi0_opt)/(spd_V + wind_x1_init*cos(psi0_opt) + wind_x2_init*sin(psi0_opt))];

	[t_sim_traj, q_sim_traj] = ode45(@system_dynamics, ...
		linspace(0, tf_opt, 1E3), q_init, ode_solver_options);
	
	solution_found_ = 1;

	if check_h
		hamiltonian_traj= zeros(numel(t_sim_traj), 1);
		tr2_sim			= zeros(numel(t_sim_traj), 1);
		for m1 = 1:numel(t_sim_traj)
			x1_t	= q_sim_traj(m1, 1);
			x2_t	= q_sim_traj(m1, 2);
			psi_t	= q_sim_traj(m1, 3);
			p1_t	= q_sim_traj(m1, 4);
			p2_t	= q_sim_traj(m1, 5);

			[wind_x1_t, wind_x2_t, ~, ~, ~, ~] = wind_fcn(x1_t, x2_t, wind_fcn_params);

			tr2_sim(m1) = abs(tan(psi_t) - p2_t/p1_t);

			hamiltonian_traj(m1) = 1 + p1_t*(spd_V*cos(psi_t) + wind_x1_t) + ...
				p2_t*(spd_V*sin(psi_t) + wind_x2_t);
		end
	end
	
	%----- Check if Hamiltonian is practically zero
	if ~check_h || (check_h && ...
			(max(abs(hamiltonian_traj)) < hamiltonian_tol && tf_opt > 0) )
		%----- Local extremum found
		cost_traj	= tf_opt;
% 		if (cost_traj < min_cost) || ...
% 				(norm(q_sim_traj(end, 1:2) - x_term') < min_bd_tol)
		if (norm(q_sim_traj(end, 1:2) - x_term') < min_bd_tol)
			min_cost	= cost_traj;
			min_bd_tol	= norm(q_sim_traj(end, 1:2) - x_term');
			y_star		= y_opt;
			solution_found_ = 1;
			
			optimal_solutions = [optimal_solutions [y_opt; cost_traj]];
		end
	end
end

%% Simulate system with optimal control (sanity check)
if solution_found_ || just_sim
% 	psi0_opt= initial_guess_opt_psi0;
% 	tf_opt	= initial_guess_opt_tf;
	
	if just_sim
		psi0_opt	= solver_params.psi0;
		tf_opt		= solver_params.tf;
	else
		psi0_opt	= y_star(1);
		tf_opt		= y_star(2);
	end

	q_init		= [x_init; psi0_opt; ...
		-cos(psi0_opt)/(spd_V + wind_x1_init*cos(psi0_opt) + wind_x2_init*sin(psi0_opt)); ...
		-sin(psi0_opt)/(spd_V + wind_x1_init*cos(psi0_opt) + wind_x2_init*sin(psi0_opt))];
	
	[t_sim_traj, q_sim_traj, ~, ~, ~]= ...
		ode45(@system_dynamics, linspace(0, tf_opt, 100), q_init, ode_solver_options);
	n_sim_pts	= length(t_sim_traj);
	
	sim_result_	= [x_init; x_term; y_star; min_cost; n_sim_pts;
		t_sim_traj; q_sim_traj(:)];
	return;
else
	sim_result_ = [];
end
	
%%	STATE, CONTROL, AND COSTATE DYNAMICS ALONG OPTIMAL TRAJECTORIES
	function q_dot = system_dynamics(t_, q_)
		
		q_dot	= zeros(5, 1);
		
		x1_	= q_(1);
		x2_	= q_(2);
		psi_= q_(3);
		p1_	= q_(4);
		p2_	= q_(5);
		
		[wind_x1_, wind_x2_, ...
			wind_gradient_11_, wind_gradient_12_, ...
			wind_gradient_21_, wind_gradient_22_] = ...
			wind_fcn(x1_, x2_, wind_fcn_params);
		q_dot(1:2,1)= spd_V*[cos(psi_); sin(psi_)] + [wind_x1_; wind_x2_];	% Aircraft kinematics
		
		q_dot(3, 1)	= wind_gradient_21_*(sin(psi_)^2) + ...
			(wind_gradient_11_ - wind_gradient_22_)*sin(psi_)*cos(psi_) - ...
			wind_gradient_12_*(cos(psi_)^2);								% Optimal control law
		
		q_dot(4:5,1)= -[wind_gradient_11_ wind_gradient_21_; ...
			wind_gradient_12_ wind_gradient_22_]*[p1_; p2_];				% Costate dynamics
	end

%%	STATE AND CONTROL DYNAMICS ALONG OPTIMAL TRAJECTORIES (W/O COSTATE)
	function q_dot = state_dynamics(t_, q_)
		
		q_dot	= zeros(3, 1);
		x1_	= q_(1);
		x2_	= q_(2);
		psi_= q_(3);
		
		[wind_x1_, wind_x2_, ...
			wind_gradient_11_, wind_gradient_12_, ...
			wind_gradient_21_, wind_gradient_22_] = ...
			wind_fcn(x1_, x2_, wind_fcn_params);
		q_dot(1:2)	= spd_V*[cos(psi_); sin(psi_)] + [wind_x1_; wind_x2_]; % Aircraft kinematics
		
		q_dot(3)	= wind_gradient_21_*(sin(psi_)^2) + ...
			(wind_gradient_11_ - wind_gradient_22_)*sin(psi_)*cos(psi_) - ...
			wind_gradient_12_*(cos(psi_)^2);								% Optimal control law
	end

%%	BOUNDARY CONDITIONS
	function f_	= boundary_conditions(y_)
		
		psi0_	= y_(1);
		tf_		= y_(2);
		
		q_init_	= [x_init; psi0_];
		
		[~, q_sim] = ode45(@state_dynamics, [0 tf_], q_init_, ode_solver_options);
		position_error = q_sim(end, 1:2)' - x_term;
		f_	= position_error;
	end

%% WORKSPACE BOUNDARY EVENT DETECTOR
	function [value_, is_terminal_, direction_] = wksp_bd_cross_event(t_, q_)
		value_			= [...
			solver_params.x1_max - q_(1); q_(1) - solver_params.x1_min; ...
			solver_params.x2_max - q_(2); q_(2) - solver_params.x2_min];
		is_terminal_	= ones(4, 1);
		direction_		= -1*ones(4, 1);
	end 
end

