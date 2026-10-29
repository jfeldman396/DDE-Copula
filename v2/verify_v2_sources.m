function manifest=verify_v2_sources
% Check published source identities without executing estimator code.
root=fileparts(mfilename('fullpath'));repo=fileparts(root);
manifest=jsondecode(fileread(fullfile(root,'provenance','SOURCE_MANIFEST.json')));
for k=1:numel(manifest.files)
 entry=manifest.files(k);file=fullfile(repo,entry.path);
 fid=fopen(file,'rb');assert(fid>=0,'V2:MissingSource','Missing %s',entry.path);
 bytes=fread(fid,Inf,'*uint8');fclose(fid);
 digest=java.security.MessageDigest.getInstance('SHA-256');digest.update(bytes);
 value=lower(reshape(dec2hex(typecast(digest.digest(),'uint8'),2).',1,[]));
 assert(strcmp(value,entry.sha256)&&numel(bytes)==entry.bytes,'V2:SourceChanged','Changed %s',entry.path);
end
end
