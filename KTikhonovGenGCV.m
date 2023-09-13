function [x,mu]=KTikhonovGenGCV(A,b,k,L)
% Solves the Tikhonov problem in general form
% x=argmin ||Ax-b||^2+mu*||Lx||
% in the GK Krylov subspace of dimension k 
% Determining mu with the GCV

[~,B,V] = lanc_b(A,b(:),k);
e=zeros(2*k+1,1);
e(1)=norm(b(:));
lv=L*V(:,1);
LV=zeros(length(lv),k);
LV(:,1)=lv;
for j=1:k
    LV(:,j)=L*V(:,j);
end
[~,R]=qr(LV,0);

mu=gcv(full(B),R,e(1:k+1));

y=[B;sqrt(mu)*R]\e;

x=V*y;
end

function mu=gcv(A,L,b)
[U,~,~,S,La] = gsvd(A,L);
bhat=U'*b;
l=diag(La);
s=diag(S);
extreme=1;
M=1e2;
while extreme
    mu=fminbnd(@gcv_funct,0,M,[],s,l,bhat(1:length(s)));
    if abs(mu-M)/M<1e-3
        M=M*100;
    else
        extreme=0;
    end
    if M>1e10
        extreme=0;
    end
end
end

function G=gcv_funct(mu,s,l,bhat)
num=(l.^2.*bhat./(s.^2+mu*l.^2)).^2;
num=sum(num);
den=(l.^2./(s.^2+mu*l.^2));
den=sum(den)^2;
G=num/den;
end


function [U,B,V] = lanc_b(A,p,k)
N=numel(p);
M=numel(A'*p);
U = zeros(N,k+1);
V = zeros(M,k);
B = sparse(k+1,k);
% Prepare for Lanczos iteration.
v = zeros(M,1);
beta = norm(p);
if (beta==0)
    error('Starting vector must be nonzero')
end
u = p/beta;
U(:,1) = u;
% Perform Lanczos bidiagonalization with/without reorthogonalization.
for i=1:k
    r=A'*u;
    r = r - beta*v;
    for j=1:i-1
        r = r - (V(:,j)'*r)*V(:,j);
    end
    alpha = norm(r);
    v = r/alpha;
    B(i,i) = alpha;
    V(:,i) = v;
    p=A*v;
    p = p - alpha*u;
    for j=1:i
        p = p - (U(:,j)'*p)*U(:,j);
    end
    beta = norm(p);
    u = p/beta;
    B(i+1,i) = beta;
    U(:,i+1) = u;
end

end
