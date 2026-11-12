function [b1,b2]=revision_align_poisson(b1,b2,truth)
% Align padded loading matrices without indexing an unpadded probability vector.
k1=size(b1,2)-1;k2=size(b2,2)-1;
assert(isequal(size(b1),size(truth.B1_unscaled))&&isequal(size(b2),size(truth.B2)));
cost=zeros(k1);for k=1:k1,cost(k,:)=-sum(b1(truth.B1_unscaled(:,k+1)>0,2:end),1);end
[a1,~]=munkres(cost);assert(isequal(sort(a1(:))',(1:k1)));
b1=[b1(:,1),b1(:,a1+1)];b2=b2(a1,:);
cost=zeros(k2);for k=1:k2,cost(k,:)=-sum(b2(truth.B2(:,k+1)>0,2:end),1);end
[a2,~]=munkres(cost);assert(isequal(sort(a2(:))',(1:k2)));b2=[b2(:,1),b2(:,a2+1)];
end
