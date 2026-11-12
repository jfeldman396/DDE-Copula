function s=dc_state(f,state1,state2,active1,active2)
for key={'B1','B2','A1','A2','Z','gamma','prop','theta','theta2'}
    s.(key{1})=f.(key{1});
end
s.state1=state1; s.state2=state2; s.active1=active1; s.active2=active2;
end
