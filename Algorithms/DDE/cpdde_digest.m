function h=cpdde_digest(parts)
% SHA256 of explicitly separated arrays/text, in MATLAB column-major order.
% NaN payloads are normalized before hashing; source manifests bind this rule.
md=java.security.MessageDigest.getInstance('SHA-256');
for i=1:numel(parts)
 p=parts{i};
 if isnumeric(p)||islogical(p)
  tag=unicode2native([class(p),':',mat2str(size(p)),':'],'UTF-8');
  if isfloat(p),p(isnan(p))=NaN;end
  if islogical(p),bytes=uint8(p(:));else,bytes=typecast(p(:),'uint8');end
 else
  text=jsonencode(p);tag=unicode2native(['json:',num2str(numel(text)),':'],'UTF-8');
  bytes=unicode2native(text,'UTF-8');
 end
 md.update(uint8(tag));if ~isempty(bytes),md.update(uint8(bytes));end;md.update(uint8(0));
end
h=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2)',1,[]));
end
