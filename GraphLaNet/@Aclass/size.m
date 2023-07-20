function [M,N]=size(A,k)
if nargin==1
    if strcmp(A.bc,'none')
        N=A.n*A.m;
        z=zeros(A.n,A.m);
        r=A.radius;
        z=z(r+1:end-r,r+1:end-r);
        M=numel(z);
    else
        N=A.n*A.m;
        M=A.n*A.m;
    end
else
    if k~=1 && k~=2
        error('k should be either 1 or 2')
    else
        N=A.n*A.m;
        M=A.n*A.m;
    end
end