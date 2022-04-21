
function [Y_label, iter_num, obj] = DBSC(B,W, label,c)
% Input
% B m*n anchor graph
% label is initial label n*1
% c is the number of clusters
% W=B'*B similarity matrix with n*n size
% Output
% Y_label is the label vector n*1
% iter_num is the number of iteration
% obj is the objective function value


[~,n] = size(B);
Y = sparse(1:n,label,1,n,c,n);  % transform label into indicator matrix 
last = 0;
iter_num = 0;
%% compute Initial objective function value
D=eye(n,n);
obj(1)= trace(Y'*(eye(n)-W)*Y*(Y'*eye(n)*Y)^(-1));
%% store once
for i=1:n
    XX(i)=B(:,i)'* B(:,i);
end    
BB = B*Y;
aa=sum(Y,1);% diag(Y'*Y) ;
YBBY=BB'*BB;% Y'*X'*X*Y;
while any(label ~= last)   
    last = label;       
 for i = 1:n   
     m = label(i) ;
    if aa(m)==1
        continue;  
    end 
    for k = 1:c        
        if k == m   
           V1(k) = YBBY(k,k)- 2 * B(:,i)'* BB(:,k) + XX(i);
           delta(k) = YBBY(k,k) / aa(k) - V1(k) / (aa(k) -1); 
        else  
           V2(k) =(YBBY(k,k)  + 2 * B(:,i)'* BB(:,k) + XX(i));
           delta(k) = V2(k) / (aa(k) +1) -  YBBY(k,k)  / aa(k); 
        end         
    end  
    [~,q] = max(delta);     
    if m~=q        
         BB(:,q)=BB(:,q)+B(:,i); % BB(:,p)=B*Y(:,p);
         BB(:,m)=BB(:,m)-B(:,i); % BB(:,m)=B*Y(:,m);
         aa(q)= aa(q) +1; %  YY(p,p)=Y(:,p)'*Y(:,p);
         aa(m)= aa(m) -1; %  YY(m,m)=Y(:,m)'*Y(:,m)
         YBBY(m,m)=V1(m); 
         YBBY(q,q)=V2(q);
         label(i)=q;
    end
 end 
  iter_num = iter_num+1;
%% compute objective function value
Y = sparse(1:n,label,1,n,c,n);  % transform label into indicator matrix ;
obj(iter_num+1) = trace(Y'*(D-W)*Y*(Y'*D*Y)^(-1)) ;      
end    
Y_label=label;
