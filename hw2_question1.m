%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%EECE5644 Fall 2023
%Homework #2
%Question #1
%Code quoted from Google Drive exercise materials
%g/practice/EMforGMM.m
%g/practice/evalGaussian.m
%g/practice/generateDataFromGMM.m
%g/practice/randGaussian.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;
dimension = 2; % For a 2-dimensional real-valued random vector X
D.d20.N = 20;
D.d200.N = 200;
D.d2k.N = 2000;
D.d10k.N = 10e3;
dTypes = fieldnames(D);
p = [0.6, 0.4]; % class priors
% class 0
mu0 = [-1 1; -1 1];
Sigma0(:,:,1) = [1 0; 0 1];
Sigma0(:,:,2) = [1 0; 0 1];
alpha0 = [1/2 1/2];
% class 1
mu1 = [-1 1; 1 -1];
Sigma1(:,:,1) = [1 0; 0 1];
Sigma1(:,:,2) = [1 0; 0 1];
alpha1 = [1/2 1/2];
figure;
%Generate Data
for ind=1:length(dTypes)
D.(dTypes{ind}).x=zeros(dimension,D.(dTypes{ind}).N); %Initialize Data
%Determine Posteriors
D.(dTypes{ind}).labels = rand(1,D.(dTypes{ind}).N)>=p(1);
D.(dTypes{ind}).N0=sum(~D.(dTypes{ind}).labels);
D.(dTypes{ind}).N1=sum(D.(dTypes{ind}).labels);
D.(dTypes{ind}).phat(1)=D.(dTypes{ind}).N0/D.(dTypes{ind}).N;
D.(dTypes{ind}).phat(2)=D.(dTypes{ind}).N1/D.(dTypes{ind}).N;
[D.(dTypes{ind}).x(:,~D.(dTypes{ind}).labels),...
D.(dTypes{ind}).dist(:,~D.(dTypes{ind}).labels)]=...
randGMM(D.(dTypes{ind}).N0,alpha0,mu0,Sigma0);
[D.(dTypes{ind}).x(:,D.(dTypes{ind}).labels),...
D.(dTypes{ind}).dist(:,D.(dTypes{ind}).labels)]=...
randGMM(D.(dTypes{ind}).N1,alpha1,mu1,Sigma1);
subplot(2,2,ind);
plot(D.(dTypes{ind}).x(1,~D.(dTypes{ind}).labels),...
D.(dTypes{ind}).x(2,~D.(dTypes{ind}).labels),'b.','DisplayName','Class 0');
hold all;
plot(D.(dTypes{ind}).x(1,D.(dTypes{ind}).labels),...
D.(dTypes{ind}).x(2,D.(dTypes{ind}).labels),'r.','DisplayName','Class 1');
grid on;
xlabel('x1');ylabel('x2');
title([num2str(D.(dTypes{ind}).N) ' Samples From Two Classes']);
end
legend 'show';
px0=evalGMM(D.d10k.x,alpha0,mu0,Sigma0);
px1=evalGMM(D.d10k.x,alpha1,mu1,Sigma1);
discScore=log(px1./px0);
sortDS=sort(discScore);
%Generate vector of gammas for parametric sweep
logGamma=[min(discScore)-eps sort(discScore)+eps];
prob=CalcProb(discScore,logGamma,D.d10k.labels,D.d10k.N0,D.d10k.N1,D.d10k.phat);
logGamma_ideal=log(p(1)/p(2));
decision_ideal=discScore>logGamma_ideal;
p10_ideal=sum(decision_ideal==1 & D.d10k.labels==0)/D.d10k.N0;
p11_ideal=sum(decision_ideal==1 & D.d10k.labels==1)/D.d10k.N1;
pFE_ideal=(p10_ideal*D.d10k.N0+(1-p11_ideal)*D.d10k.N1)/(D.d10k.N0+D.d10k.N1);
%Estimate Minimum Error
%If multiple minimums are found choose the one closest to the theoretical
%minimum
[prob.min_pFE, prob.min_pFE_ind]=min(prob.pFE);
if length(prob.min_pFE_ind)>1
[~,minDistTheory_ind]=min(abs(logGamma(prob.min_pFE_ind)-logGamma_ideal));
prob.min_pFE_ind=prob.min_pFE_ind(minDistTheory_ind);
end
%Find minimum gamma and corresponding false and true positive rates
minGAMMA=exp(logGamma(prob.min_pFE_ind));
prob.min_FP=prob.p10(prob.min_pFE_ind);
prob.min_TP=prob.p11(prob.min_pFE_ind);
%Plot
plotROC(prob.p10,prob.p11,prob.min_FP,prob.min_TP);
hold all;
plot(p10_ideal,p11_ideal,'+','DisplayName','Ideal Min. Error');
plotMinPFE(logGamma,prob.pFE,prob.min_pFE_ind);
plotDecisions(D.d10k.x,D.d10k.labels,decision_ideal);
plotERMContours(D.d10k.x,alpha0,mu0,Sigma0,alpha1,mu1,Sigma1,logGamma_ideal);
fprintf('Theoretical: Gamma=%1.2f, Error=%1.2f%%\n',...
 exp(logGamma_ideal),100*pFE_ideal);
fprintf('Estimated: Gamma=%1.2f, Error=%1.2f%%\n',minGAMMA,100*prob.min_pFE);
for ind=1:length(dTypes)-1
%Estimate Parameters using matlab built in function
D.(dTypes{ind}).DMM_Est0=...
fitgmdist(D.(dTypes{ind}).x(:,~D.(dTypes{ind}).labels)',2,'Replicates',10);
D.(dTypes{ind}).DMM_Est1=...
fitgmdist(D.(dTypes{ind}).x(:,~D.(dTypes{ind}).labels)',1,'Replicates',10);
plotContours(D.(dTypes{ind}).x,...
D.(dTypes{ind}).DMM_Est0.ComponentProportion,...
D.(dTypes{ind}).DMM_Est0.mu,D.(dTypes{ind}).DMM_Est0.Sigma);
%Calculate discriminate score
px0=pdf(D.(dTypes{ind}).DMM_Est0,D.d10k.x');
px1=pdf(D.(dTypes{ind}).DMM_Est1,D.d10k.x');
discScore=log(px1'./px0');
sortDS=sort(discScore);
%Generate vector of gammas for parametric sweep
logGamma=[min(discScore)-eps sort(discScore)+eps];
prob=CalcProb(discScore,logGamma,D.d10k.labels,...
D.d10k.N0,D.d10k.N1,D.(dTypes{ind}).phat);
%Estimate Minimum Error
%If multiple minimums are found choose the one closest to the theoretical
%minimum
[prob.min_pFE, prob.min_pFE_ind]=min(prob.pFE);
if length(prob.min_pFE_ind)>1
[~,minDistTheory_ind]=...
min(abs(logGamma(prob.min_pFE_ind)-logGamma_ideal));
prob.min_pFE_ind=min_pFE_ind(minDistTheory_ind);
end
%Find minimum gamma and corresponding false and true positive rates
minGAMMA=exp(logGamma(prob.min_pFE_ind));
prob.min_FP=prob.p10(prob.min_pFE_ind);
prob.min_TP=prob.p11(prob.min_pFE_ind);
%Plot
plotROC(prob.p10,prob.p11,prob.min_FP,prob.min_TP);
plotMinPFE(logGamma,prob.pFE,prob.min_pFE_ind);
fprintf('Estimated: Gamma=%1.2f, Error=%1.2f%%\n',...
minGAMMA,100*prob.min_pFE);
end
options=optimset('MaxFunEvals',3000,'MaxIter',1000);
for ind=1:length(dTypes)
lin.x=[ones(1,D.(dTypes{ind}).N); D.(dTypes{ind}).x];
lin.init=zeros(dimension+1,1);
% [lin.theta,lin.cost]=thetaEst(lin.x,lin.init,D.(dTypes{ind}).labels);
[lin.theta,lin.cost]=...
fminsearch(@(theta)(costFun(theta,lin.x,D.(dTypes{ind}).labels)),...
lin.init,options);
lin.discScore=lin.theta'*[ones(1,D.d10k.N); D.d10k.x];
gamma=0;
lin.prob=CalcProb(lin.discScore,gamma,D.d10k.labels,...
D.d10k.N0,D.d10k.N1,D.d10k.phat);
% quad.decision=[ones(D.d10k.N,1) D.d10k.x]*quad.theta>0;
plotDecisions(D.d10k.x,D.d10k.labels,lin.prob.decisions);
title(sprintf('Data and Classifier Decisions Against True Label for Linear Logistic Fit\nProbability of Error=%1.1f%%',100*lin.prob.pFE));
% plotDecisions(D.d10k.x,D.d10k.labels,quad.decision);
quad.x=[ones(1,D.(dTypes{ind}).N); D.(dTypes{ind}).x;...
D.(dTypes{ind}).x(1,:).^2;...
D.(dTypes{ind}).x(1,:).*D.(dTypes{ind}).x(2,:);...
D.(dTypes{ind}).x(2,:).^2];
quad.init= zeros(2*(dimension+1),1);
[quad.theta,quad.cost]=...
fminsearch(@(theta)(costFun(theta,quad.x,D.(dTypes{ind}).labels)),...
quad.init,options);
quad.xScore=[ones(1,D.d10k.N); D.d10k.x; D.d10k.x(1,:).^2;...
D.d10k.x(1,:).*D.d10k.x(2,:); D.d10k.x(2,:).^2];
quad.discScore=quad.theta'*quad.xScore;
gamma=0;
quad.prob=CalcProb(quad.discScore,gamma,D.d10k.labels,...
D.d10k.N0,D.d10k.N1,D.d10k.phat);
plotDecisions(D.d10k.x,D.d10k.labels,quad.prob.decisions);
title(sprintf('Data and Classifier Decisions Against True Label for Linear Logistic Fit\nProbability of Error=%1.1f%%',100*quad.prob.pFE));
end
