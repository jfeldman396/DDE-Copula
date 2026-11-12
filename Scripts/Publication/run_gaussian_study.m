function run_gaussian_study(data_input,output_root)
% Explicit opt-in reproduction of the original fixed4k+4k protocol.
% No raw survey input is distributed; output_root must be new.
assert(nargin==2 && isfile(data_input));assert(~isfolder(output_root),'Choose a new output directory.');
md=java.security.MessageDigest.getInstance('SHA-256');fid=fopen(data_input,'rb');assert(fid>0);cleanup=onCleanup(@()fclose(fid));
while ~feof(fid),md.update(fread(fid,1048576,'*uint8'));end
input_hash=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));clear cleanup;
assert(strcmp(input_hash,'0b2848f2875580338b4fc71eccd696c0fadb69b5d2c712859f10255c5210233f'),'External input hash differs from the archived factor-study input.');
here=fileparts(mfilename('fullpath'));repo=fileparts(fileparts(here));
addpath(fullfile(repo,'Algorithms','GaussianCopula'));
addpath(fullfile(repo,'Utilities','GaussianCopula'));
plan=fullfile(repo,'Results','Publication','data','factor','PLAN.json');
mkdir(output_root);four=fullfile(output_root,'stage4k');eight=fullfile(output_root,'stage8k');mkdir(four);mkdir(eight);
gaussian_review(true,data_input,four,plan);
gaussian_review(false,data_input,four,plan);
copyfile(fullfile(four,'PREFLIGHT.json'),fullfile(eight,'PREFLIGHT.json'));
for id=1:6
 folder=fullfile(eight,sprintf('c%d',id));mkdir(folder);
 for c=1:2
  filename=sprintf('chain%d.mat',c);copyfile(fullfile(four,sprintf('c%d',id),filename),fullfile(folder,filename));
 end
end
gaussian_review_8k(false,data_input,eight,four,plan);
end
