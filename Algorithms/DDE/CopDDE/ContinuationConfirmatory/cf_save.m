function cf_save(path,record)
tmp=[path '.tmp.mat']; save(tmp,'record','-v7.3'); movefile(tmp,path,'f');
end
