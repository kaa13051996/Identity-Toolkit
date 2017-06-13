%% used for searching weights of different gmms. 
%best result was using ga from toolbox, not this.
p0=linspace(-1,1,10);
p1=linspace(-1,1,10);
p2=linspace(-1,1,10);
p3=linspace(-1,1,10);
[F,S,F1,S1] = ndgrid(p0, p1,p2,p3);
[fitresult,~,~] = arrayfun(@(pp0,pp1,pp2,pp3) compute_eer(pp0*scores0+pp1*scores1+pp2*scores2+pp3*scores3, labels, false), F, S, F1,S1); 
[minval, minidx] = min(fitresult(:));
bestFirst = F(minidx);
bestSecond = S(minidx);
disp(minval);

%%
fun = @(x) compute_eer(x(1)*scores0+x(2)*scores1+x(3)*scores2+x(4)*scores3, labels, false);
%@(x) compute_eer(x(1)*scores0+x(2)*scores_lda, labels, false);
%gfun = @(x) deal(g(x(1),x(2)),[]);
x0 = [1;0;1;0];
%options = optimoptions('fminunc','Algorithm','quasi-newton');
options = optimoptions('fmincon','Algorithm','interior-point','Display','iter');
%options.Display = 'iter';
%[x, fval, exitflag, output] = fminunc(fun,x0,options);
%[x,fval,exitflag,output] = fmincon(fun,x0,[],[],[],[],[],[],gfun,options);

%save('E:\temp\123\data\DBN\scores.mat','scores0','scores1','scores2','scores3','scores_ada1','scores_ada2','scores_ada3','scores_lda1','scores_lda2','scores_lda3','scores_tree1','scores_tree2','scores_tree3');
%save('E:\temp\123\data\DBN\scores2.mat','scores0','scores1','scores2','scores3','scores_ada1','scores_ada2','scores_ada3','scores_lda1','scores_lda2','scores_lda3','scores_tree1','scores_tree2','scores_tree3','scores_nb2','scores_nb3','scores_svml2','scores_svml3');
%save('E:\temp\123\data\DBN\scores2.mat','scores0','scores1','scores2','scores3','scores_ada1','scores_ada2','scores_ada3','scores_lda1','scores_lda2','scores_lda3','scores_tree1','scores_tree2','scores_tree3','scores_nb1','scores_nb2','scores_nb3','scores_svml1','scores_svml2','scores_svml3');

%Z = [scores0,scores1,scores2,scores3,scores_ada1,scores_ada2,scores_ada3,scores_lda1,scores_lda2,scores_lda3,scores_svml2,scores_svml3];
%X=pinv(Z'*Z)*(Z'*labels)
%normirovka
%minVal = min(Z,[],1);
%maxVal = max(Z,[],1);
%Z1=(Z-repmat(minVal, [length(Z),1]))./repmat(maxVal-minVal,[length(Z),1]);
%normalization
%ZMean = repmat(mean(Z),length(Z),1);
%ZVariance = repmat(var(Z),length(Z),1);
%Zn = (Z - ZMean)./(ZVariance);
%compute_eer(X(1)*scores0+X(2)*scores1+X(3)*scores2+X(4)*scores3+X(5)*scores_ada1+X(6)*scores_ada2+X(7)*scores_ada3+X(8)*scores_lda1+X(9)*scores_lda2+X(10)*scores_lda3+X(11)*scores_svml2+X(12)*scores_svml3, labels, true);
%@(X) compute_eer(X(1)*scores0+X(2)*scores1+X(3)*scores2+X(4)*scores3+X(5)*scores_ada1+X(6)*scores_ada2+X(7)*scores_ada3+X(8)*scores_lda1+X(9)*scores_lda2+X(10)*scores_lda3+X(11)*scores_svml2+X(12)*scores_svml3, labels,false)
%[0 0 0 0 0 0 0 0 0 0 0 0]
%@(X) compute_eer(X(1)*scores0+X(2)*scores1+X(3)*scores2+X(4)*scores3+X(5)*scores_ada1+X(6)*scores_ada2+X(7)*scores_ada3+X(8)*scores_lda1+X(9)*scores_lda2+X(10)*scores_lda3+X(11)*scores_svml2+X(12)*scores_svml3+X(13)*scores_tree1+X(14)*scores_tree2+X(15)*scores_tree3+X(16)*scores_nb2+X(17)*scores_nb3, labels,false)
%@(X) compute_eer(X(1)*Zn(:,1)+X(2)*Zn(:,2)+X(3)*Zn(:,3)+X(4)*Zn(:,4)+X(5)*Zn(:,5)+X(6)*Zn(:,6)+X(7)*Zn(:,7)+X(8)*Zn(:,8)+X(9)*Zn(:,9)+X(10)*Zn(:,10)+X(11)*Zn(:,11)+X(12)*Zn(:,12), labels,false)