%init param
%synaptic weights
W11 = 0.8616;
W12 = 0.0198;
W21 = 0.1360;
W22 = 0.6185;
%bias
b11 = 0.7261;
b12 = 0.4453;
b2  = 0.2882;
bias = 0.1960;
%error init
iter = 0 ;
error = 1 ;
%input
input = 0.4332;
%target output
Target = 1.707
while 1==1
    iter = iter + 1
    if abs(error) <= 0.001
        break
    else
        %forward
        N11 = W11*input+b11*bias
        A11 = logsig(N11)
        N12 = W12*input+b12*bias
        A12 = logsig(N12)
        A2 = purelin((W21*A11+W22*A12)+b2*bias)
        error = Target-A2
        %Backpropagation
        S2 =-2*1*error
        S11 = (1-A11)*A11*(W21*S2)
        S12 = (1-A12)*A12*(W22*S2)
        %update param
        alpha = 0.1
        W11 = W11-alpha*S11*input
        b11 = b11-alpha*S11
        W12 = W12-alpha*S12*input
        b12 = b12-alpha*S12
        W21 = W21-alpha*S2*A11
        b2  = b2-alpha*S2
        W22 = W22-alpha*S2*A12
    end
end
