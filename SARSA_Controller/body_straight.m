function [te,xe] = body_straight(t, x0, u, par)
    % calculate the outcome state and time for one step   
    x = ode2(@equations_of_motion_train,t,x0,par,u);
    te = t(end);
    xe = x(end,:);
end


 