"""Fixed-parameter ordinal DDE prediction. No fitting and no test-set access.

Exact marginal cutpoints enumerate only each item's nonzero parent support.
Exact conditional prediction enumerates all shallow states up to a declared
budget; larger models use separately validated conditional binary Gibbs.
Gamma is a variance. Input categories for score_rows are 1..5, missing NaN.
"""
from dataclasses import dataclass
import json, time
from pathlib import Path
import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtr
from scipy.optimize import brentq


def log_interval(lo, hi):
    lo, hi = np.broadcast_arrays(np.asarray(lo, float), np.asarray(hi, float))
    if np.isnan(lo).any() or np.isnan(hi).any() or np.any(lo > hi):
        raise ValueError('Invalid normal interval')
    ans = np.full(lo.shape, -np.inf)
    neg = (hi <= 0) & (lo < hi)
    pos = (lo >= 0) & (lo < hi)
    cross = (lo < 0) & (hi > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        a, b = log_ndtr(hi[neg]), log_ndtr(lo[neg])
        ans[neg] = a + np.log(-np.expm1(b-a))
        a, b = log_ndtr(-lo[pos]), log_ndtr(-hi[pos])
        ans[pos] = a + np.log(-np.expm1(b-a))
        ans[cross] = np.log(ndtr(hi[cross])-ndtr(lo[cross]))
    narrow = np.isfinite(lo) & np.isfinite(hi) & (lo < hi)
    narrow &= (hi-lo)*(1+np.maximum(np.abs(lo), np.abs(hi))) < .01
    if narrow.any():
        x,w=np.polynomial.legendre.leggauss(8)
        half=(hi[narrow]-lo[narrow])/2;mid=lo[narrow]+half
        ans[narrow]=np.log(half)-.5*np.log(2*np.pi)+logsumexp(np.log(w)[None,:]-.5*(mid[:,None]+half[:,None]*x)**2,axis=1)
    return ans


def states(k, start=0, stop=None):
    if stop is None: stop=1<<k
    return ((np.arange(start,stop,dtype=np.uint64)[:,None] >> np.arange(k,dtype=np.uint64)) & 1).astype(float)


def validate_response(y,hidden,J):
    y=np.asarray(y,float);hidden=np.asarray(hidden,bool)
    if y.shape!=(J,) or hidden.shape!=(J,):raise ValueError('Response/mask shape')
    known=np.isfinite(y)
    if np.isinf(y).any() or not np.isin(y[known],[1,2,3,4,5]).all():raise ValueError('Expected integer categories1..5 or NaN')
    if (hidden&~known).any():raise ValueError('Cannot mask an originally missing outcome')
    return y,hidden


@dataclass
class DDEModel:
    B1: np.ndarray
    B2: np.ndarray
    gamma: np.ndarray
    prop: np.ndarray

    def __post_init__(self):
        self.B1=np.asarray(self.B1,float);self.B2=np.asarray(self.B2,float)
        self.gamma=np.asarray(self.gamma,float).ravel();self.prop=np.asarray(self.prop,float).ravel()
        self.J,self.K1=self.B1.shape[0],self.B1.shape[1]-1
        self.K2=len(self.prop)
        if self.K1 < 1 or self.B2.shape != (self.K1,self.K2+1): raise ValueError('DDE dimensions')
        if self.gamma.shape != (self.J,) or np.any(self.gamma<=0): raise ValueError('Positive residual variances required')
        if not all(np.isfinite(x).all() for x in (self.B1,self.B2,self.gamma,self.prop)):raise ValueError('Nonfinite parameters')
        if np.any((self.prop<0)|(self.prop>1)): raise ValueError('Invalid Bernoulli probabilities')

    def lower_logprior(self, aa, subset=None):
        b=self.B2 if subset is None else self.B2[np.asarray(subset,dtype=int)]
        top=states(self.K2)
        with np.errstate(divide='ignore'):
            ltop=np.where(top==1,np.log(self.prop),np.log1p(-self.prop)).sum(axis=1)
        logits=b[:,0][None,:]+top@b[:,1:].T
        l0=-np.logaddexp(0,logits);l1=-np.logaddexp(0,-logits)
        # a*l1+(1-a)*l0 avoids subtracting infinities for deterministic priors.
        terms=[]
        for u in range(len(top)):
            terms.append(ltop[u]+(np.where(aa==1,l1[u],l0[u])).sum(axis=1))
        return logsumexp(np.asarray(terms),axis=0)


def load_legacy_endpoint(path):
    """Load a selected saved fit on a copy; never use its training latent rows."""
    import h5py
    with h5py.File(path,'r') as h:
        g=h['endpoint/f']
        b1=g['B1'][()].T;b2=g['B2'][()].T
        a1=g['active1'][()].ravel().astype(bool);a2=g['active2'][()].ravel().astype(bool)
        b1[:,np.flatnonzero(~a1)+1]=0;b2[:,np.flatnonzero(~a2)+1]=0
        s=np.flatnonzero(np.any(b1[:,1:]!=0,axis=0))
        u=np.flatnonzero(np.any(b2[s,1:]!=0,axis=0))
        return DDEModel(b1[:,np.r_[0,s+1]],b2[s][:,np.r_[0,u+1]],g['gamma'][()].ravel(),g['prop'][()].ravel()[u])


def calibrate(model,cdf_edges):
    """Exact model-CDF cutpoints, exploiting item-specific sparse support."""
    q=np.asarray(cdf_edges,float)
    if q.shape!=(model.J,6) or np.any(np.diff(q,axis=1)<=0) or np.any(q[:,0]!=0) or np.any(q[:,-1]!=1): raise ValueError('Ordered full-support CDF edges required')
    thresholds=np.empty(q.shape);thresholds[:,0]=-np.inf;thresholds[:,-1]=np.inf
    errors=[];support_sizes=[]
    for j in range(model.J):
        support=np.flatnonzero(model.B1[j,1:]!=0);support_sizes.append(len(support))
        if len(support)>24:raise ValueError('Exact marginal support budget exceeded; no silent approximation')
        means=[];weights=[]
        for start in range(0,1<<len(support),65536):
            a=states(len(support),start,min(start+65536,1<<len(support)))
            means.append(model.B1[j,0]+a@model.B1[j,support+1])
            weights.append(model.lower_logprior(a,support))
        means=np.concatenate(means);lp=np.concatenate(weights);lp-=logsumexp(lp);w=np.exp(lp)
        sd=np.sqrt(model.gamma[j]);lo=means.min()-12*sd;hi=means.max()+12*sd
        def cdf(z,survival=False):
            v=(z-means)/sd
            return float(w@ndtr(-v if survival else v))
        for c in range(1,5):
            target=q[j,c]
            f=(lambda z:cdf(z)-target) if target<=.5 else (lambda z:cdf(z,True)-(1-target))
            z=brentq(f,lo,hi,xtol=2e-12,rtol=8*np.finfo(float).eps)
            thresholds[j,c]=z;errors.append(abs(cdf(z)-target))
    return thresholds,dict(max_marginal_cdf_error=max(errors),item_parent_support=support_sizes,method='exact sparse item-parent enumeration')


class ExactPredictor:
    def __init__(self,model,thresholds,max_states=65536):
        if (1<<model.K1)>max_states: raise ValueError('Exact state budget exceeded')
        self.model=model;self.thresholds=np.asarray(thresholds,float)
        a=states(model.K1);self.logprior=model.lower_logprior(a);self.logprior-=logsumexp(self.logprior)
        mu=model.B1[:,0]+a@model.B1[:,1:].T;sd=np.sqrt(model.gamma)
        self.emissions=log_interval((self.thresholds[None,:,:-1]-mu[:,:,None])/sd[None,:,None],(self.thresholds[None,:,1:]-mu[:,:,None])/sd[None,:,None])

    def predict(self,y,hidden,identity=0):
        y,hidden=validate_response(y,hidden,self.model.J)
        observed=np.flatnonzero(np.isfinite(y)&~hidden);targets=np.flatnonzero(np.isfinite(y)&hidden)
        if not len(targets):raise ValueError('No masked outcomes')
        c=np.where(np.isfinite(y),y-1,0).astype(int)
        lw=self.logprior+self.emissions[:,observed,c[observed]].sum(axis=1)
        normalizer=logsumexp(lw)
        if not np.isfinite(normalizer):raise ArithmeticError('Conditioning event has zero or nonfinite model probability')
        lw-=normalizer
        cat=logsumexp(lw[:,None,None]+self.emissions[:,targets,:],axis=0)
        actual=cat[np.arange(len(targets)),c[targets]]
        joint=float(logsumexp(lw+self.emissions[:,targets,c[targets]].sum(axis=1)))
        if np.max(np.abs(np.exp(cat).sum(1)-1))>1e-10:raise ArithmeticError('Category normalization')
        return dict(targets=targets,log_probability=actual,all_log_probabilities=cat,log_joint=joint,method='exact',integration_diagnostic=0.)


class ChunkedExactPredictor:
    """Memory-bounded exact reference for occasional large-K validation rows."""
    def __init__(self,model,thresholds,chunk=16384,max_states=1<<22):
        if (1<<model.K1)>max_states:raise ValueError('Chunked exact work budget exceeded')
        self.model=model;self.thresholds=np.asarray(thresholds,float);self.chunk=chunk

    def predict(self,y,hidden,identity=0):
        m=self.model;y,hidden=validate_response(y,hidden,m.J)
        obs=np.flatnonzero(np.isfinite(y)&~hidden);targets=np.flatnonzero(np.isfinite(y)&hidden)
        c=np.where(np.isfinite(y),y-1,0).astype(int);t=self.thresholds
        if not len(targets):raise ValueError('No masked outcomes')
        den=-np.inf;num=np.full((len(targets),5),-np.inf);joint=-np.inf
        for start in range(0,1<<m.K1,self.chunk):
            a=states(m.K1,start,min(start+self.chunk,1<<m.K1))
            mu=m.B1[:,0]+a@m.B1[:,1:].T;sd=np.sqrt(m.gamma)
            lw=m.lower_logprior(a)+log_interval((t[obs,c[obs]]-mu[:,obs])/sd[obs],(t[obs,c[obs]+1]-mu[:,obs])/sd[obs]).sum(axis=1)
            e=log_interval((t[targets,:-1][None,:,:]-mu[:,targets,None])/sd[None,targets,None],(t[targets,1:][None,:,:]-mu[:,targets,None])/sd[None,targets,None])
            den=np.logaddexp(den,logsumexp(lw))
            num=np.logaddexp(num,logsumexp(lw[:,None,None]+e,axis=0))
            joint=np.logaddexp(joint,logsumexp(lw+e[:,np.arange(len(targets)),c[targets]].sum(axis=1)))
        if not np.isfinite(den):raise ArithmeticError('Impossible conditioning event')
        cat=num-den
        if np.max(abs(np.exp(cat).sum(1)-1))>1e-10:raise ArithmeticError('Category normalization')
        return dict(targets=targets,log_probability=cat[np.arange(len(targets)),c[targets]],all_log_probabilities=cat,log_joint=float(joint-den),method='exact_chunked',integration_diagnostic=0.)


def score_rows(predictor,Y,masks,row_ids,output_csv=None):
    import pandas as pd
    rows=[];start=time.perf_counter()
    for r in range(len(masks)):
        for i in range(len(Y)):
            p=predictor.predict(Y[i],masks[r,i],identity=int(row_ids[i])*10+r);lp=p['log_probability']
            rows.append(dict(row_id=int(row_ids[i]),mask_repeat=r,masked_count=len(lp),negative_log_probability_sum=float(-lp.sum()),joint_negative_log_probability=-p['log_joint'],method=p['method'],integration_diagnostic=p['integration_diagnostic'],integration_pass=p.get('integration_pass',True),max_rhat=p.get('max_rhat',1.),draws_per_chain=p.get('draws_per_chain',0)))
    df=pd.DataFrame(rows)
    if output_csv:Path(output_csv).parent.mkdir(parents=True,exist_ok=True);df.to_csv(output_csv,index=False)
    return df,dict(conditional_log_loss=float(df.negative_log_probability_sum.sum()/df.masked_count.sum()),joint_masked_log_loss=float(df.joint_negative_log_probability.mean()),joint_per_masked_item=float(df.joint_negative_log_probability.sum()/df.masked_count.sum()),respondents=len(Y),mask_repeats=len(masks),integration_pass=bool(df.integration_pass.all()),integration_failures=int((~df.integration_pass).sum()),elapsed_seconds=time.perf_counter()-start)
