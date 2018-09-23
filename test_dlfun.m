clc
close all
clear all


%% Load Data
load('Datasets/MNSIT_2digits_100.mat');
load('Datasets/Exact_20bases_MNSIT_2digits_100.mat');
X=X';

[N,L] = size(X); 
K=20;
nofIt = 10;

dlsODL = struct('D',X, 'Met','ODL', 'vsMet','ORMP', 'vsArg',struct('tnz',4));
dlsODL.lamStep = 20; dlsODL.ndStep = 50;
dlsODL.A = eye(K); dlsODL.B = dlsODL.D;
dlsODL.snr = zeros(1,nofIt);          

tic; 
for i=1:nofIt 
    dlsODL.lam = lambdafun(i, 'C', nofIt, 0.99, 1); 
    dlsODL = dlfun(dlsODL, X, 1); 
end
toc;  

sigma=0.25;
