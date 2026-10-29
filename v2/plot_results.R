args <- commandArgs(trailingOnly=TRUE)
p <- normalizePath(if(length(args)) args[1] else 'v2/evidence')
if(length(args)<2) stop('Supply a NEW figure output directory as the second argument.')
out <- args[2]
if(dir.exists(out)||file.exists(out)) stop('Output directory already exists; refusing overwrite.')
dir.create(out,recursive=TRUE)
s <- read.csv(file.path(p,'primary_tempering_CP.csv'))
m <- read.csv(file.path(p,'matched_three_methods.csv'))
cols <- c(H1='#176B87',H0='#A65E00',Z='#963D83'); syms <- c(H1=16,H0=17,Z=1)
ns <- c(500,1000,2000,4000,8000,16000)
render <- function(name,fun,w=12,h=10){
 png(file.path(out,paste0(name,'.png')),width=w,height=h,units='in',res=160);fun();dev.off()
 pdf(file.path(out,paste0(name,'.pdf')),width=w,height=h,useDingbats=FALSE);fun();dev.off()
}
legendtop <- function(title,foot){mtext(title,outer=TRUE,side=3,line=2,cex=1.1,font=2);mtext('H1: maxima + full averaging',outer=TRUE,side=3,line=.6,adj=.06,col=cols['H1'],cex=.8);mtext('H0: maxima + raw objectives',outer=TRUE,side=3,line=.6,adj=.5,col=cols['H0'],cex=.8);mtext('Z: Gibbs + full averaging',outer=TRUE,side=3,line=.6,adj=.94,col=cols['Z'],cex=.8);mtext(foot,outer=TRUE,side=1,line=1,cex=.75)}
render('primary_all_fit_errors_and_recovery',function(){
 par(mgp=c(3,.6,0),cex.axis=.85,mfrow=c(3,3),mar=c(4.2,5.2,3.3,1),oma=c(3,0,4,0))
 for(metric in c('B1_MSE_all','B2_MSE_all','joint_correct')) for(J in c(50,100,150)){
  d <- s[s$J==J,]; yr<-range(d[[metric]]);isrec<-metric=='joint_correct'
  plot(1,1,type='n',xlim=c(1,6),ylim=if(isrec)c(0,100) else yr,log=if(isrec)'' else 'y',xaxt='n',xlab='Sample size',ylab=if(isrec)'Joint recovery / 100' else paste(sub('_MSE_all','',metric),'all-fit MSE'),main='',las=1)
  title(main=paste('J =',J),line=.8)
  axis(1,1:6,c('500','1k','2k','4k','8k','16k'))
  for(method in names(cols)){z<-d[d$method==method,];z<-z[order(z$n),];lines(1:6,z[[metric]],col=cols[method],pch=syms[method],type='b',lty=match(method,names(cols)),lwd=1.5)}
 }
 legendtop('Primary results under tempering and CP','100 datasets per cell. Every fit retained. MSE axes are logarithmic; padding differs across J. Lines show means, not uncertainty.')
})
render('matched_conditional_errors',function(){
 par(mgp=c(3,.6,0),cex.axis=.85,mfrow=c(2,3),mar=c(4.2,5.2,3.3,1),oma=c(4,0,4,0))
 for(metric in c('B1_MSE','B2_MSE')) for(J in c(50,100,150)){
  d<-m[m$J==J&m$scope=='per_n',]
  plot(1,1,type='n',xlim=c(1,6),ylim=range(d[[metric]]),log='y',xaxt='n',xlab='Sample size',ylab=paste(sub('_MSE','',metric),'matched conditional MSE'),main='',las=1)
  title(main=paste('J =',J),line=1.7)
  axis(1,1:6,c('500','1k','2k','4k','8k','16k'))
  for(method in names(cols)){z<-d[d$method==method,];z<-z[order(z$n),];lines(1:6,z[[metric]],col=cols[method],pch=syms[method],type='b',lty=match(method,names(cols)),lwd=1.5)}
  z<-d[d$method=='H1',];z<-z[order(z$n),];mtext(paste('Eligible N:',paste(z$N,collapse=', ')),side=3,line=.2,cex=.73)
 }
 legendtop('Same correct-dimension datasets across all three methods','Tempering + CP. Eligibility is per sample size; all methods must recover both dimensions. No outlier exclusions.\nConditional selection is not overall performance. Means shown; paired uncertainty is in the source PAIRED table.')
},12,8.5)
render('runtime_and_cap',function(){
 par(mgp=c(3,.6,0),cex.axis=.85,mfrow=c(2,3),mar=c(4.2,5.2,3.3,1),oma=c(3,0,4,0))
 for(metric in c('median_CPU_s','cap_count')) for(J in c(50,100,150)){
  d<-s[s$J==J,];cap<-metric=='cap_count'
  plot(1,1,type='n',xlim=c(1,6),ylim=if(cap)c(0,100) else range(d[[metric]]),xaxt='n',xlab='Sample size',ylab=if(cap)'Reached cap / 100' else 'Median inference CPU (s)',main='',las=1)
  title(main=paste('J =',J),line=.8)
  axis(1,1:6,c('500','1k','2k','4k','8k','16k'))
  for(method in names(cols)){z<-d[d$method==method,];z<-z[order(z$n),];lines(1:6,z[[metric]],col=cols[method],pch=syms[method],type='b',lty=match(method,names(cols)),lwd=1.5)}
 }
 legendtop('Runtime and iteration cap under tempering and CP','CPU includes inherited prefix once per arm; not batch cost. Timings mix execution epochs. Cap and stabilization can overlap.')
},12,8.5)
