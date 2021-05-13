function TD = controller(x,par,Q)
    s = discretize_state(x, par);
    a = execute_policy(Q, s);
    TD = take_action(a, par);
end

function s = discretize_state(x, par)
    %  Discretize state. 
    s = [0,0,0,0];
    pd = linspace(0, 2*pi, par.pos_states);
    vd = linspace(-pi, pi, par.vel_states);
    
    % The angle should be transformed into [0,pi]
    ps1 = mod(x(1),2*pi);ps2 = mod(x(2),2*pi);
    [~,Index1] = min(abs(pd-ps1));[~,Index2] = min(abs(pd-ps2));
    s(1) = Index1;s(2) = Index2;
    
    % Cut the velocity to be in the given range.
    vs1 = cut(x(3));vs2 = cut(x(4));
    [~,Index3] = min(abs(vd-vs1));[~,Index4] = min(abs(vd-vs2));
    s(3) = Index3;s(4) = Index4;
end

function vs = cut(x)
    if x>pi
        vs = pi;
    elseif x<-pi
        vs = -pi;
    else
        vs = x;
    end
end

function u = take_action(a, par)
    % Calculate the proper torque for action a
    ud = linspace(-par.maxtorque,par.maxtorque,par.actions);
    u = ud(a);
end

function a = execute_policy(Q, s)
    % Select an action for state s 
    [~,idx] = max(Q(s(1),s(2),s(3),s(4),:));
    a = idx;
end