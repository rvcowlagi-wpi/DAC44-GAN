clear variables; close all; clc
addpath(genpath('My-Classes'))

%----- Problem dimensions
N_THREAT_STATE	= 9;
N_GRID_ROW		= 5;
SENSOR_NOISE_VAR= 0.1;	% Variance of (i.i.d.) measurement noise in each sensor, assuming homogeneous sensors


grid_		= ACEGridWorld(1, N_GRID_ROW);
threat_		= ParametricThreat(N_THREAT_STATE, ...
	grid_.halfWorkspaceSize, SENSOR_NOISE_VAR, grid_);
trueThreat_k	= threat_.calculate_at_locations( ...
		grid_.coordinates(:, sensor_.configuration) );

flags_.SHOW_TRUE	= true;
flags_.SHOW_ESTIMATE= false;
flags_.JUXTAPOSE	= true;
flags_.SHOW_PATH	= true;
flags_.DUAL_SCREEN	= true;
grid_.plot_parametric(threat_, [], flags_)