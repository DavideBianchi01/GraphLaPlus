function [x,V,k,M]=l2lqDPRestarted(A,b,L,q,e,noise_norm,tau,iter,tol,Rest)
delta=tau*noise_norm;
b=b(:);
%u=memory;
%base=u.MemUsedMATLAB;
% waitbar
h=waitbar(0,'l_p-l_q - Computations in progress...');

% Initializations
x=A'*b;
M=zeros(iter+1,1);
% Creating initial space
v=x;
nv=norm(v(:));
V=v(:)/nv;
AV=A*V;

% Creating L matrix
LV=L*V;

% Inital QR factorization
[QA,RA]=qr(AV,0);
[QL,RL]=qr(LV,0);

% Compute regularization parameter
%u=memory;
%M(1)=u.MemUsedMATLAB-base;


u=L*x;
y=nv;
% Begin MM iterations
for k=1:iter
    if mod(k,Rest)==0
        x=V*y;
        %clear V AV LV QA RA QR RL
        clear V AV LV QA RA QR RL
        V=x/norm(x);
        %V=x(:)/norm(x(:));
        AV=A*V;
        LV=L*V;
        [QA,RA]=qr(AV,0);
        [QL,RL]=qr(LV,0);
    end
    % Store previous iteration for stopping criterion
    y_old=y;
    
    % Compute weights for approximating p/q norms with the 2 norm
    try
    wr=u(:).*(1-((u(:).^2+e^2)/e^2).^(q/2-1));
    catch
        disp('error')
    end
    % Solve re-weighted linear system
    c=e^(q-2);
    eta=discrepancyPrinciple(delta,RA,RL,QA,QL,b,wr,c);
    %modificato da me
    eta=max(eta,1e-6);
    %----------------
    y=[RA; sqrt(eta)*RL]\[QA'*(b(:)); sqrt(eta)*(QL'*(wr))];
    %uM=memory;
    %M(k+1)=uM.MemUsedMATLAB-base;
    % Check stopping criteria
    %if k>1 && numel(y)>1 && norm(y-[y_old;0],'fro')/norm([y_old;0],'fro')<tol
      %  M=M(1:k+1);
     %   break     
    %end
    if k<iter && mod(k+1,Rest)~=0
        v=AV*y-b;
        u=LV*y;
        % Compute residual
        ra=v;
        ra=A'*ra;
        rb=(u-wr(:));
        rb=L'*rb;
        r=ra(:)+eta*rb(:);
        r=r-V*(V'*r);
        r=r-V*(V'*r);
        % Enlarge space and update QR factorizations
        [AV,LV,QA,RA,QL,RL,V]=updateQR(A,L,AV,LV,QA,RA,QL,RL,V,r);
    end
    waitbar(k/(iter-1));
end
x= V*y;
try
    close(h)
catch
    warning('Waitbar not closed')
end
end


function mu=discrepancyPrinciple(delta,RA,RL,QA,QL,b,wr,c)
mu=1e-30;
[U,V,~,C,S]=gsvd(RA,RL);
what=V'*QL'*wr;
bhat=QA'*b;
bb=b-QA*bhat;
nrmbb=norm(bb);
bhat=U'*bhat;
a=diag(C);
l=diag(S);
for i=1:30
    mu_old=mu;
    f=((c*a.*l.*what-c*bhat.*l.^2).^2)'*((mu*a.^2+c*l.^2).^-2)-delta^2+nrmbb^2;
    fprime=-2*(a.^2.*(c*a.*l.*what-c*bhat.*l.^2).^2)'*((mu*a.^2+c*l.^2).^-3);
    mu=mu-f/fprime;
    if abs(mu_old-mu)/mu_old<1e-6
        break
    end
end
mu=c/mu;
end

function [AV,LV,QA,RA,QL,RL,V]=updateQR(A,L,AV,LV,QA,RA,QL,RL,V,r)
vn=r/norm(r(:));
Avn=A*vn;
AV=[AV, Avn(:)];
Lvn=L*vn;
LV=[LV, Lvn(:)];
V=[V,vn];
rA=QA'*Avn(:);
qA=Avn(:)-QA*rA;
tA=norm(qA(:));
qtA=qA/tA;
QA=[QA,qtA];
RA=[RA rA;zeros(1,length(rA)) tA];
rL=QL'*Lvn(:);
qL=Lvn(:)-QL*rL;
tL=norm(qL(:));
qtL=qL/tL;
QL=[QL,qtL];
RL=[RL rL;zeros(1,length(rL)) tL];
end