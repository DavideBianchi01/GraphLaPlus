function [L,W,D,A] = computeL(I,sigmaInt,R)
% Input:  - I ,image matrix  
%         - sigmaInt, parameter of the weight function
%         - R, neighborhood radius in infinity norm ( R=2 è R=1 nella
%         pratica)

if nargin<3
    R=3;
end
sigmaDist=100;
R=R-1;
[nr,nc]=size(I);
n=nr*nc; %n° nodes of the graph = n° of pixels
% vecI=I(:); %gives an order to the pixels


k=1; iW=zeros((2*R+1)^2*n,1); jW=zeros((2*R+1)^2*n,1); vW=zeros((2*R+1)^2*n,1);
for x1=1:nc
for y1=1:nr
    for x2=max(x1-R,1):min(x1+R,nc)
    for y2=max(y1-R,1):min(y1+R,nr)
        node1=(y1-1)*nr+x1; node2=(y2-1)*nr+x2; %sorting of the pixels
        if x1~=x2 || y1~=y2
            dist=I(x1,y1)-I(x2,y2);
            iW(k)=node1; jW(k)=node2; 
            vW(k)=exp(-dist^2/sigmaInt);%exp(-norm([x1;y1]-[x2;y2])^2/sigmaDist);
            k=k+1;
        end
    end
    end
end
end
iW=iW(1:k-1); jW=jW(1:k-1); vW=vW(1:k-1);
W=sparse(iW,jW,vW,n,n);
A=W;
W=W./norm(W(:));

d=sum(W);
D=spdiags(d',0,n,n);
L=D-W;

end

