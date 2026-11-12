function render_nofloor_weight_review(out)
status=jsondecode(fileread(fullfile(out,'STATUS.json')));notice='';if status.preliminary,notice='PRELIMINARY: incomplete study. ';end;s=load(fullfile(out,'matrices.mat'));ns=[500 1000 2000 4000 8000 16000];
for d=2:3
 lim=0;errlim=0;for n=ns,lim=max(lim,max(abs(s.(sprintf('B%d_n%d_truth',d,n))),[],'all'));errlim=max(errlim,max(abs(s.(sprintf('B%d_n%d_error',d,n))),[],'all'));end
 f=figure('Visible','off','Color','w','Position',[100 100 2200 1000]);tiledlayout(3,6,'TileSpacing','compact','Padding','compact');
 for row=1:3
  kinds={'mean','truth','error'};labels={'Mean estimate','Mean matched truth','Mean signed error'};
  for n=ns
   ax=nexttile;B=s.(sprintf('B%d_n%d_%s',d,n,kinds{row}));cl=lim;if row==3,cl=errlim;end
   imagesc(B,[-cl cl]);colormap(ax,parula(256));set(ax,'FontName','Arial','FontSize',12,'XColor','k','YColor','k','Color','w');
   if row==1,title(sprintf('n=%d',n),'Color','k');end
   if n==500,ylabel(labels{row},'Color','k');end
   if row==3,xlabel('Column (including intercept)','Color','k');end
   if n==16000,cb=colorbar;cb.Color='k';end
  end
 end
 sgtitle(sprintf('%sLayer %d: gap+5, no-floor maxima + tempering + CAVI; all dimensions correct',notice,d),'Color','k','FontName','Arial');
 exportgraphics(f,fullfile(out,sprintf('B%d_estimates.png',d)),'Resolution',160);exportgraphics(f,fullfile(out,sprintf('B%d_estimates.pdf',d)),'ContentType','vector');close(f);
end
end
