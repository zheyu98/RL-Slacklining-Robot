function x0 = initial_state()
    phi0 = pi/2; %(rad)
    delta0 = pi/2 + pi/1000;
    x0 =[delta0; phi0; zeros(2,1)];
end
