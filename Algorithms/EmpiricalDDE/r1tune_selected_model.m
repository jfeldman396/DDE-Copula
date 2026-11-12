function model=r1tune_selected_model(raw)
% Match the published estimator's final CSP column pruning on a copy.
% The raw optimization checkpoint and all latent/parameter updates are intact.
model=raw;
if isfield(raw,'predictive_pruning'),return;end
if isfield(raw,'B') && iscell(raw.B)
 D=numel(raw.B);before=zeros(1,D);after=before;
 for d=1:D
  assert(size(raw.B{d},2)==numel(raw.active{d})+1);
  before(d)=sum(any(raw.B{d}(:,2:end)~=0,1));
  model.B{d}(:,find(~raw.active{d})+1)=0;
  after(d)=sum(any(model.B{d}(:,2:end)~=0,1));
 end
else
 assert(isfield(raw,'B1') && isfield(raw,'B2'));
 before=[sum(any(raw.B1(:,2:end)~=0,1)),sum(any(raw.B2(:,2:end)~=0,1))];
 assert(size(raw.B1,2)==numel(raw.active1)+1 && size(raw.B2,2)==numel(raw.active2)+1);
 model.B1(:,find(~raw.active1)+1)=0;model.B2(:,find(~raw.active2)+1)=0;
 after=[sum(any(model.B1(:,2:end)~=0,1)),sum(any(model.B2(:,2:end)~=0,1))];
end
model.predictive_pruning=struct('rule','zero CSP-inactive loading columns at return, as published',...
 'raw_nonzero_widths',before,'selected_nonzero_widths',after);
end
