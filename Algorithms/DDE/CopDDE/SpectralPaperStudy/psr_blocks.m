function b=psr_blocks(J,K1,K2)
% Width-adaptive partitions: about3 layer1 blocks, joint small top layer.
assert(J>=K1&&K1>=3&&K2==floor(K1/3));
b1=min(8,ceil(K1/3));
if K2<=6,b2=K2;else,b2=min(8,ceil(K2/2));end
b=[b1 b2];
end
