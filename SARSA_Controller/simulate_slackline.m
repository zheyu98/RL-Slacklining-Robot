%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Parameters:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load 'parameters.mat' 'Q';
par = robot_set_parameters;
par.pos_states = 7;     % Position discretization
par.vel_states = 7;     % Velocity discretization
par.actions = 5;        % Action discretization
%visualization choices:
showanimation=1;%do you want to see an animation? Then set this to 1. 

%simulation parameters:
Ts=.0001;%sampling rate (s)
tend=4;%end time of simulation (s)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%initial conditions for the simulation:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%initial segment angles:
phi0 = pi/2+pi/500;%(rad)
delta0 = pi/2 + pi/500;
%initial slackline 'radius':
r = par.r;%m

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%prepare simulation:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%initial time and state:
t0 = 0;
x0 =[delta0; phi0; zeros(2,1)];

%prepare parameter structure to pass to equations of motion (eom):
m = par.m;
l = par.l;
I = par.I;
c = par.c;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%simulate:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

state = step(tend,x0,Ts,par,Q);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%extract and process data from the simulation output:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%extract states from output:
delta = state(:,1); phi = state(:,2);
ddelta = state(:,3); dphi = state(:,4);

%store for visualization:
footx=r.*cos(delta);
footy=-r.*sin(delta);

hipx=r.*cos(delta)+l.*cos(phi);
hipy=-r.*sin(delta)+l.*sin(phi);

%plot joint angles:
figure(1)
clf
t = [0:Ts:tend];
plot(t,phi*180/pi);
hold on
plot(t,delta*180/pi);
legend('phi','gamma');
ylabel('angles in deg')
xlabel('time in s')
title(gca,'Angles against Time (Small initial deviations)')
ylim([80,100]);
set(gcf,'name','angles')
hold off

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%animate, if desired:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if showanimation
    %prepare visualization:
    figure(2)
    clf
    
    stanceleghandle=plot([0,0],[-r,-r+l],'k','linewidth',3);
    hold on
    slackhandle=plot([0,0],[0,-r]);
    
    xlim([-1,1])
    ylim([-1,1])
    
    xlabel('x in m')
    ylabel('y in m')
    
    set(gcf,'name','slackliner animation')
    
    % Record the animation
    writerObj=VideoWriter('Simple model_small.avi');  
    open(writerObj);
    
    %loop through the data to update the animated figure:
    for index=1:100:tend/Ts-1
        set(slackhandle,'XData',[0,footx(index)],'YData',[0,footy(index)])
        set(stanceleghandle,'XData',[footx(index),hipx(index)],'YData',[footy(index),hipy(index)])          
        shg
        frame = getframe;
        frame.cdata = imresize(frame.cdata, [543 429]); 
        writeVideo(writerObj,frame);
    end
    close(writerObj);
end
%% Function for numerical integration
function XV = step(t,xv0, dt, par, Q)
    n = round(t/dt);
    XV = zeros(n+1,4);
    XV(1,:) = xv0; 
    for i = 1:n
        d = equations_of_motion(n,XV(i,:),par,Q);
        XV(i+1,:) = XV(i,:)+d*dt;
    end
end




