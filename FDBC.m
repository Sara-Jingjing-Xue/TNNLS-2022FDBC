
function [U_label, iter_num, obj_max] = FDBC(B, label,c)
% Input
% B m*n anchor graph (Z in the paper)
% label is initial label n*1
% c is the number of clusters
% Output
% U_label is the label vector n*1
% iter_num is the number of iteration
% obj_max is the objective function value
% It is written by Jingjing Xue 
% code for "Fast Clustering by Directly Solving Bipartite Graph Clustering Problem" in TNNLS


[~,n] = size(B);
U = sparse(1:n,label,1,n,c,n);  % transform label into indicator matrix 
last = 0;
iter_num = 0;

%% store once
aa=sum(U,1);
[~,label]=max(U,[],2);
BBB=2*(B'*B);
XX=diag(BBB)./2;

BBUU= BBB* U;% BBUU(i,:) 
ybby=diag(U'*BBUU/2);
%% compute Initial objective function value
obj_max(1) = sum(ybby./aa') ; 
while any(label ~= last)   
    last = label;       
 for i = 1:n   
     m = label(i) ;
    if aa(m)==1
        continue;  
    end 
        V21=ybby'+(BBUU(i,:)+XX(i)).*(1-U(i,:));
        V11=ybby'-(BBUU(i,:)-XX(i)).*U(i,:);
        delta= V21./(aa+1-U(i,:))-V11./(aa-U(i,:));  
    [~,q] = max(delta);     
    if m~=q                
        aa(q)= aa(q) +1; %  YY(p,p)=Y(:,p)'*Y(:,p);
        aa(m)= aa(m) -1; %  YY(m,m)=Y(:,m)'*Y(:,m)    
        ybby(m)=V11(m); %
        ybby(q)=V21(q);
        U(i,m)=0;
        U(i,q)=1;
        label(i)=q;
        BBUU(:,m)=BBUU(:,m)-BBB(:,i);% 
        BBUU(:,q)=BBUU(:,q)+BBB(:,i);    
    end          
 end 
  iter_num = iter_num+1;
%% compute objective function value
obj_max(iter_num+1) = sum(ybby./aa') ; % max
end    
U_label=label;
