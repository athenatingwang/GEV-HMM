library(HiddenMarkov)
library(extRemes)


## run HMM
load(file="../MaximaData/R1wkmax.image")
R1hrmax <- R1wkmax

#nk = as.integer(system('echo $SK', intern=T)) # code for supercomputers
nk <- 5  #number of states

#kk = as.integer(system('echo $KK', intern=T)) ## number of runs, code for supercomputers

source("hmm.evd.R")


kk <- 601
catched <- FALSE 
#while(!catched) 
while(kk <= 2000)
  {
  temp <- matrix(runif(nk*nk,0,1),ncol=nk)
  lam <- rgamma(1,3,1)
  diag(temp) = diag(temp) + rpois(1,lam) * apply(temp, 1, sum)
  temp <- temp * matrix(rep(1/apply(temp, 1, sum), ncol(temp)), ncol=ncol(temp), byrow=F)
  Pit <- log(temp/diag(temp))
  Pi<-as.vector(Pit[!diag(nk)])
  min.x <- min(R1hrmax)
  max.x <- max(R1hrmax)
  n<-length(R1hrmax)
  
  delta <- c(2,runif(nk-1, 0,1))
  delta <- delta/sum(delta)
  
  loct <- sort(runif(nk,0,10))
  shapet <- sort(runif(nk,-2,5))
  scalet <- sort(runif(nk,0,10))
  param0<-c(loct,log(scalet),shapet,Pi)
  
  
  tryCatch({
    y <- nlm(hmm.evd,p=param0,data=R1hrmax,nk=nk,iterlim = 10000,steptol = 1e-4,print.level=2)
    loc.est <- y$estimate[1:(nk)]
    scale.est <- exp(y$estimate[(nk+1):(2*nk)])
    shape.est <- y$estimate[(2*nk+1):(3*nk)]
    
    tgamman<-exp(y$estimate[(3*nk+1):length(y$estimate)])
    tgamma<-diag(nk)
    
    tgamma[!tgamma]<-tgamman
    gamman<-y$estimate[(3*nk+1):(length(y$estimate)+nk)]<-tgamma/apply(tgamma,1,sum)
    
    delta<-solve(t(diag(nk)-gamman+1),rep(1,nk))
    
    HMMest <- list(loct=loc.est, scale=scale.est, shape=shape.est, Pi=gamman,delta=delta, LL=-y$minimum)
    nLL <- -y$minimum
    
    if (abs(nLL)<=52115){  
      catched = TRUE
      eval(parse(text=paste('HMM',nk,'est = HMMest',sep="")))
      eval(parse(text=paste('save(HMM',nk,'est, file="HMMgevWk',nk,'est',kk,'.image")',sep='')))
      kk <- kk + 1
    }
  }, error=function(e){})
}




