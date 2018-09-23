clc
close all
clear all

% addpath 'nmflib/'
addpath(genpath('utils'));  % add K-SVD box
addpath(genpath('LCKSVD'));  % add K-SVD box
addpath('COPAR');
addpath('DLSI');
addpath('LRSDL_FDDL');
addpath('build_spams');
addpath('utils');
addpath('SRC');
addpath('ODL');

%% Load Data
load('Datasets/MNSIT_2digits_100.mat');
load('Datasets/Exact_20bases_MNSIT_2digits_100.mat');
X=X';

%% Create noisy data
A=[];
sigma=0.25;
for l=1:10
    X1=X+sigma*abs(randn(size(X)));  %--Absolute Gaussian noise
   %X1=X+mat2gray(poissrnd(sigma,size(X))); %--Poisson Noise
    A=[A;X1];
%     A=[A;X];
end

figure();
one_piece_of_2_digits = reshape(X(1,:),[],56);
imshow(one_piece_of_2_digits);

sigma=0.5;

% [acc_lrsdl, rt] = LRSDL_wrapper(Y_train, label_train, Y_test , label_test, ...
%                             k, k0, lambda1, lambda2, lambda3);

opts.verbal=1;
opts.verbose=0;
opts.max_iter      = 500;
% function [D, X] = ODL(Y, k, lambda, opts, method)                       
[D, H] = ODL(A, 20, sigma, opts, 'spams'); % fista spams
opts.holdsD = D;
Dopt = D;
for i=1:20
    [D, H] = ODL(A, 20, sigma, opts, 'spams'); % fista spams
    opts.holdsD = D;
    Dopt = D;
end    

figure()
for i=1:20
    subplot(5,4,i)
%      H11(i,:)=mat2gray(H1(i,:));
%      imshow((reshape(H11(i,:),[],56)));
     imshow(reshape(H(i,:),[],56))
end
sigma=0.25;

%


%


%


%

