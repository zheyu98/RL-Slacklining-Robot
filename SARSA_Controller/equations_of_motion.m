function [statederivative]=equations_of_motion(t0,state,par,Q)
%extract parameters:
g = par.g; % gravity (m/s^2)
m = par.m; % mass (kg)
I = par.I; % inertia (kgm^2)
l = par.l; % length of the pendulum (m)
r = par.r; % slackline deformation (m)
c = par.c; % stiffness of slackline (N/m)

%state vector:
%delta,phi,ddelta,dphi
%with:
gamma = state(1);
phi = state(2); 
%derivatives:
dgamma = state(3);
dphi = state(4); 

%----------------------------------------------------------------------
%INPUTS （torque from the flywheel)
%----------------------------------------------------------------------

% Get torque from the controller
tau = controller(state,par,Q);

%%
%----------------------------------------------------------------------
%EQUATIONS OF MOTION (DERIVED VIA TMT):
%----------------------------------------------------------------------
%mass matrix:
Mij = diag([m,m,I]);
Tij = [-r*sin(gamma), -l/2*sin(phi);
    -r*cos(gamma), l/2*cos(phi);
    0,1];
Fi = [-r*c*cos(gamma); r*c*sin(gamma)-m*g; tau];
gk = [-r*cos(gamma)*dgamma^2-l/2*cos(phi)*dphi^2;
    r*sin(gamma)*dgamma^2-l/2*sin(phi)*dphi^2;
    0];
M_bar = Tij.'*Mij*Tij;
Q_bar = Tij.'*(Fi-Mij*gk);
accs = M_bar\Q_bar;
statederivative = [state(3:4), accs.'];

