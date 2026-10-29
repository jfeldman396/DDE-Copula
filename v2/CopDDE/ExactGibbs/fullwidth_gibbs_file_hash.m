function fingerprint=fullwidth_gibbs_file_hash(path)
%FULLWIDTH_GIBBS_FILE_HASH Read-only SHA-256 identity for a saved input/source.
fid=fopen(path,'rb');
assert(fid>=0,'DDE:FullwidthHashInput','Cannot read %s.',path);
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
digest=java.security.MessageDigest.getInstance('SHA-256');
byteCount=0;
while ~feof(fid)
    bytes=fread(fid,1048576,'*uint8');
    digest.update(bytes); byteCount=byteCount+numel(bytes);
end
hashed=typecast(digest.digest(),'uint8');
fingerprint=struct('sha256',lower(reshape(dec2hex(hashed,2).',1,[])), ...
    'bytes',byteCount);
end
