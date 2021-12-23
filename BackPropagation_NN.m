clear all;
clc;
%definisi iterasi
iterasi=0
%input
input=1
%output target
target=1.707
%error
error_target=0.0001
%learning rate
alpha=0.1
%weight
W11=-0.963
W12=0.643
W21=0.584
W22=0.844
%bias
bias=1
%bias weight
b11=-0.111
b12=0.231
b2=0.476
%error
error=target
while error > 0.0001
    %iterasi
    iterasi=iterasi+1;
    %forward
    N11=W11*input+b11*bias;
    OutputNode1=logsig(N11);
    N12=W12*input+b12*bias;
    OutputNode2=logsig(N12);
    OutputNode3=purelin((W21*OutputNode1+W22*OutputNode2)+b2*bias);
    %error
    error=target-OutputNode3;
    %backpropagation
    S2=-2*1*error;
    S11=(1-OutputNode1)*OutputNode1*(W21*S2);
    S12=(1-OutputNode2)*OutputNode2*(W22*S2);
    %Update parameter
    W11=W11-alpha*S11*input;
    W12=W12-alpha*S12*input;
    W21=W21-alpha*S2*OutputNode1;
    W22=W22-alpha*S2*OutputNode2;
    b11=b11-alpha*S11;
    b12=b12-alpha*S12;
    b2=b2-alpha*S2;
end
fprintf('Jumlah iterasi=%.0f\n W11=%.4f\n W12=%.4f\n W21=%.4f\n W22=%.4f\n b11=%.4f\n b12=%.4f\n b2=%.4f\n Output=%.4f\n Error=%.4f\n',iterasi, W11, W12, W21, W22, b11, b12, b2, OutputNode3, error)