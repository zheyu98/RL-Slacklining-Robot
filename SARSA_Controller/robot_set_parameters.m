function par = robot_set_parameters
%% initialize body variables
%general mechanical constants:
g=9.81;%gravity (m/s^2)

%segment properties:
m=4;%mass of segments (kg)
l=1;%length of segments (m)
I=1/12*m*(l)^2;

%slackline properties:
r=0.3;%slackline sag in static conditions (m)
c=m*g/r;%slackline stiffness, as calculated from sag (N/m)

par.g = g;
par.m = m;
par.l = l;
par.I = I;
par.r = r;
par.c = c;
par.maxtorque = 1;
