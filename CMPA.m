Gp = 0.1;
Is = 0.01e-12;
Ib = 0.1e-12;
Vb = 1.3;

V = linspace(-1.95, 0.7, 200);
I = Is.*(exp(1.2.*V./0.025)-1) + V.*Gp - Ib.*(exp(-1.2.*(V+Vb)./0.025)-1);
%Generates signal with noise as much as +/- 0.2*I
I2 = 0.8.*I + I.*0.4.*rand(1, 200);

figure(1)
plot(V, I);
hold on
plot(V, I2);
legend('Data', 'Data with noise')
title('Plot of Data and Data with added noise')
xlabel('Voltage (V)')
ylabel('Current (A)')
hold off

figure(2)
semilogy(V, I);
hold on
semilogy(V, I2);
legend('Data', 'Data with noise')
title('Plot of Data and Data with added noise with log scale current')
xlabel('Voltage (V)')
ylabel('Current (A)')
hold off

%2 ploynomial fitting

I14 = polyval(polyfit(V, I, 4), V);
I18 = polyval(polyfit(V, I, 8), V); 
I24 = polyval(polyfit(V, I2, 4), V); 
I28 = polyval(polyfit(V, I2, 8), V);  

figure(3)
plot(V, I)
hold on
plot(V, I14, '--')
plot(V, I18, '--')
legend('Data', 'poly 4', 'poly 8')
title('Polynomials fitted to data')
xlabel('Voltage (V)')
ylabel('Current (A)')
hold off

figure(4)
plot(V, I2)
hold on
plot(V, I24, '--')
plot(V, I28, '--')
legend('Data', 'poly 4', 'poly 8')
title('Polynomials fitted to noisy data')
xlabel('Voltage (V)')
ylabel('Current (A)')
hold off

% Curve fitting with fit function
fo = fittype('A.*(exp(1.2*x/25e-3)-1) + 0.1.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff = fit(V',I',fo);
If = ff(V);

figure(5)
plot(V, If)
title('Solution with Is and Id fitted')
xlabel('Voltage (V)')
ylabel('Current (A)')


fo = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff = fit(V',I',fo);
If = ff(V);

figure(6)
plot(V, If)
title('Solution with Is, Id, and Gp fitted')
xlabel('Voltage (V)')
ylabel('Current (A)')



fo = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+D))/25e-3)-1)');
ff = fit(V',I',fo);
If = ff(V);

figure(7)
plot(V, If)
title('Solution with all parameters fitted')
xlabel('Voltage (V)')
ylabel('Current (A)')

% Neural Nets
inputs = V.';
targets = I.';
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs)
view(net)
Inn = outputs