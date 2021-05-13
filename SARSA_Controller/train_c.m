function par = train_c(par)

    par.simtime = 4;     % Trial length
    par.simstep = 0.01;   % Simulation time step
    
    % Obtain SARSA parameters
    par = get_parameters(par);

    % Initialize the outer loop
    Q = init_Q(par);
    te = 0;

    % Outer loop: trials
    for ii = 1:par.trials
        % Initialize the inner loop
        x = initial_state();
        s = discretize_state(x, par);
        a = execute_policy(Q, s, par);

        % Inner loop: simulation steps
        for tt = 1:ceil(par.simtime/par.simstep)

            % obtain torque
            u = take_action(a, par);

            % Apply torque and obtain new state
            % x  : state (input at time t and output at time t+par.simstep)
            % u  : torque
            % te : new time
            [te, x] = body_straight([te te+par.simstep],x,u,par);

            % The next step
            sP = discretize_state(x, par);
            r = observe_reward(x);
            aP = execute_policy(Q, sP, par);
            Q = update_Q(Q, s, a, r, sP, aP, par);
            s = sP; a = aP;
        end
    end
    
    % save learned Q value function
%     save('parameters','Q');
end
    
function par = get_parameters(par)
    % set the values
    par.epsilon = 0;        % Random action rate
    par.gamma = 0.9;       % Discount rate
    par.alpha = 0.25;          % Learning rate
    par.pos_states = 7;     % Position discretization
    par.vel_states = 7;     % Velocity discretization
    par.actions = 5;        % Action discretization
    par.trials = 1000;         % Learning trials
end

function Q = init_Q(par)
    % Initialize the Q table.
    Q = zeros(par.pos_states,par.pos_states,par.vel_states,par.vel_states,par.actions)+5;
    t_index_p = (par.pos_states+1)/2;
    t_index_v = (par.vel_states+1)/2;
    Q(t_index_p,t_index_p,t_index_v,t_index_v,:) = 0;
end

function s = discretize_state(x, par)
    % Discretize state.
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

function r = observe_reward(x)
    % Calculate the reward for taking action a
    r = -sum((x(2)-pi/2).^2)-0.25*sum(x(4).*x(4));
end

function a = execute_policy(Q, s, par)
    % Select an action for state s using the epsilon-greedy algorithm.
    eps = rand;
    if eps>=par.epsilon
        [~,idx] = max(Q(s(1),s(2),s(3),s(4),:));
        a = idx;
    else
        a = randi(par.actions);
    end
end

function Q = update_Q(Q, s, a, r, sP, aP, par)
    % Implement the SARSA update rule.
    Q(s(1),s(2),s(3),s(4),a) = Q(s(1),s(2),s(3),s(4),a) ...
        + par.alpha*(r+par.gamma*(Q(sP(1),sP(2),sP(3),sP(4),aP)-Q(s(1),s(2),s(3),s(4),a)));
end