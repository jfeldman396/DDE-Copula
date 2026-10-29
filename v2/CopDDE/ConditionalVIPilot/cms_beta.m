function beta=cms_beta(it,slow)
paper=pcp_protocol;
if ~slow,beta=paper.beta(min(it,50));return;end
% Stretch the original beta curve from iterations1:15 to1:30.
position=1+14*min(1,(it-1)/29); beta=interp1(1:15,paper.beta(1:15),position,'linear');
end
