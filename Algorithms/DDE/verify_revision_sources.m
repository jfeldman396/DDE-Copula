function manifest=verify_revision_sources
% Verify the current reproduction sources against the single package manifest.
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
manifest=jsondecode(fileread(fullfile(root,'Results','Publication','MANIFEST.json')));
for k=1:numel(manifest.files)
 entry=manifest.files(k);[~,~,ext]=fileparts(entry.path);
 if ~ismember(ext,{'.m','.mat'}),continue;end
 file=fullfile(root,entry.path);fid=fopen(file,'rb');
 assert(fid>=0,'DDE:MissingSource','Missing %s',entry.path);
 bytes=fread(fid,Inf,'*uint8');fclose(fid);
 digest=java.security.MessageDigest.getInstance('SHA-256');digest.update(bytes);
 value=lower(reshape(dec2hex(typecast(digest.digest(),'uint8'),2).',1,[]));
 assert(strcmp(value,entry.sha256)&&numel(bytes)==entry.bytes,'DDE:SourceChanged','Changed %s',entry.path);
end
end
