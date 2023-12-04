
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
U= turnfF(label,n,c);
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
while any(label ~= last)  && iter_num < 30 
    last = label;       
 for i = 1:n   
     m = label(i) ;
    if aa(m)==1
        continue;  
    end 
        z_Y=BBUU(i,:);
        zz=XX(i);
        
        V21=ybby'+z_Y+zz;
        V21(m)=ybby(m);
        
        aa21=aa+1;
        aa21(m)=aa(m);
        
        V11=ybby';
        V11(m)=ybby(m)-z_Y(m)+zz;
        
        aa11=aa;
        aa11(m)=aa(m)-1;
        
        delta= V21./aa21-V11./aa11;        
        [~,q] = max(delta);     
    if m~=q     
        aa(q)=aa21(q);
        aa(m)=aa11(m);
        ybby(m)=V11(m); 
        ybby(q)=V21(q);
        label(i)=q;
        B_i=BBB(:,i);
        BBUU(:,m)=BBUU(:,m)-B_i; 
        BBUU(:,q)=BBUU(:,q)+B_i;    
    end          
 end 
iter_num = iter_num+1;
%% compute objective function value
obj_max(iter_num+1) = sum(ybby./aa') ; % max
end    
U_label=label;
