"""Exact sparse-support ordinal lookup prototype; independent tiny tests.

Fixed parameters and already training-calibrated cutpoints only. Constructor
precomputes the exact shallow prior and each item's local emission table.
Prediction performs lookup/summation, never a normal CDF per full state/row.
The full shallow prior is exact; no mean-field assumption or width blacklist.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math, os, sys, time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.special import logsumexp

HERE=Path(__file__).resolve().parent
from dde_predict import DDEModel,calibrate,states,log_interval,validate_response,ExactPredictor


def _hash_arrays(arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.ascontiguousarray(a)
        h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()


class SparseExactPredictor:
    def __init__(self,model,thresholds,max_states=1<<22,low_bits=16,
                 memory_budget_mb=768,prior_upper_block=16):
        started=time.perf_counter()
        # Own immutable parameter copies prevent a cached prior/emission table
        # from being used after mutation of a caller's fitted model arrays.
        self.model=DDEModel(*(np.array(x,copy=True) for x in
                           [model.B1,model.B2,model.gamma,model.prop]))
        m=self.model
        self.thresholds=np.array(thresholds,float,copy=True)
        if self.thresholds.shape!=(m.J,6) or np.isnan(self.thresholds).any():
            raise ValueError('Expected J by6 thresholds without NaNs')
        if not np.all(np.diff(self.thresholds,axis=1)>0) or not np.all(self.thresholds[:,0]==-np.inf) or not np.all(self.thresholds[:,-1]==np.inf):
            raise ValueError('Strict full-support cutpoints required')
        self.S=1<<m.K1
        if self.S>max_states: raise ValueError('Declared exact-state work budget exceeded')
        if m.K1>31: raise ValueError('Packed uint32 support index budget')
        self.low_bits=min(int(low_bits),m.K1)
        if self.low_bits<0: raise ValueError('Nonnegative low_bits required')
        self.nlo=1<<self.low_bits;self.nhi=1<<(m.K1-self.low_bits)
        self.supports=[np.flatnonzero(m.B1[j,1:]!=0) for j in range(m.J)]
        table_entries=sum((1<<len(s))*5 for s in self.supports)
        unique_supports={tuple(s) for s in self.supports}
        map_bytes=len(unique_supports)*(self.nlo+self.nhi)*4
        # Conservative live-array budget includes cached prior, four state
        # work vectors, packed query indices and bounded construction scratch.
        estimated_bytes=(table_entries*8+map_bytes+self.S*(8*5+4)+
                         self.nlo*(self.low_bits+prior_upper_block*2)*8+
                         self.nhi*(m.K1-self.low_bits+prior_upper_block)*8+
                         max((1<<len(s))*5*8 for s in self.supports)*2)
        if estimated_bytes>memory_budget_mb*1024**2:
            raise MemoryError(f'Exact lookup estimated live array budget {estimated_bytes/1024**2:.1f}MiB exceeds {memory_budget_mb}MiB')
        self.identity=_hash_arrays([m.B1,m.B2,m.gamma,m.prop,self.thresholds])
        self.logprior=self._prior(prior_upper_block)
        lp_norm=float(logsumexp(self.logprior))
        if abs(lp_norm)>2e-10: raise ArithmeticError('Exact prior normalization')
        self.logprior-=lp_norm
        maps={};self.maps=[];self.emissions=[];max_category_sum_error=0.
        for j,support in enumerate(self.supports):
            key=tuple(support)
            if key not in maps: maps[key]=self._packed_maps(support)
            self.maps.append(maps[key])
            size=1<<len(support);table=np.empty((size,5),float)
            for start in range(0,size,65536):
                stop=min(start+65536,size)
                aa=states(len(support),start,stop)
                mu=m.B1[j,0]+aa@m.B1[j,support+1]
                sd=math.sqrt(m.gamma[j]);t=self.thresholds[j]
                table[start:stop]=log_interval((t[:-1]-mu[:,None])/sd,(t[1:]-mu[:,None])/sd)
            err=float(np.max(abs(np.exp(table).sum(axis=1)-1)))
            if err>1e-10: raise ArithmeticError('Local category normalization')
            max_category_sum_error=max(max_category_sum_error,err)
            table.flags.writeable=False;self.emissions.append(table)
        for a in [m.B1,m.B2,m.gamma,m.prop,self.thresholds,self.logprior]:a.flags.writeable=False
        self.cache_diagnostic=dict(model_threshold_identity=self.identity,shallow_states=self.S,
            upper_states=1<<m.K2,low_bits=self.low_bits,high_bits=m.K1-self.low_bits,
            item_parent_support=[len(s) for s in self.supports],
            emission_table_entries=table_entries,emission_table_MiB=table_entries*8/1024**2,
            support_map_MiB=map_bytes/1024**2,prior_MiB=self.logprior.nbytes/1024**2,
            conservative_array_budget_MiB=estimated_bytes/1024**2,
            memory_budget_MiB=memory_budget_mb,prior_log_normalization_error=lp_norm,
            max_local_category_sum_error=max_category_sum_error,
            construction_seconds=time.perf_counter()-started,
            exact_prior=True,normal_CDFs_per_prediction=0)

    def _prior(self,upper_block):
        m=self.model;upper_block=int(upper_block)
        if upper_block<1: raise ValueError('Positive upper block')
        alo=states(self.low_bits);ahi=states(m.K1-self.low_bits)
        prior=np.full((self.nhi,self.nlo),-np.inf)
        for start in range(0,1<<m.K2,upper_block):
            top=states(m.K2,start,min(start+upper_block,1<<m.K2))
            with np.errstate(divide='ignore'):
                lu=np.where(top==1,np.log(m.prop),np.log1p(-m.prop)).sum(axis=1)
            logits=m.B2[:,0]+top@m.B2[:,1:].T
            base=-np.logaddexp(0,logits).sum(axis=1)+lu
            low=alo@logits[:,:self.low_bits].T+base
            high=ahi@logits[:,self.low_bits:].T
            for h in range(self.nhi):
                prior[h]=np.logaddexp(prior[h],logsumexp(low+high[h],axis=1))
        return prior.ravel()

    def _packed_maps(self,support):
        lo=np.arange(self.nlo,dtype=np.uint32);hi=np.arange(self.nhi,dtype=np.uint32)
        lomap=np.zeros(self.nlo,dtype=np.uint32);himap=np.zeros(self.nhi,dtype=np.uint32)
        for pos,k in enumerate(support):
            if k<self.low_bits:lomap|=((lo>>np.uint32(k))&np.uint32(1))<<np.uint32(pos)
            else:himap|=((hi>>np.uint32(k-self.low_bits))&np.uint32(1))<<np.uint32(pos)
        lomap.flags.writeable=False;himap.flags.writeable=False
        return lomap,himap

    def _full_index(self,j):
        lo,hi=self.maps[j]
        out=np.empty((self.nhi,self.nlo),dtype=np.uint32)
        np.add(hi[:,None],lo[None,:],out=out)
        return out.ravel()

    def _add_emission(self,weights,j,category):
        # Reuse one small low-half gather; avoid a full S byJ allocation.
        lo,hi=self.maps[j];w=weights.reshape(self.nhi,self.nlo);e=self.emissions[j]
        for h in range(self.nhi):w[h]+=e[lo+hi[h],category]

    def _stable_support_logmass(self,index,logpost,size):
        maximum=np.full(size,-np.inf)
        np.maximum.at(maximum,index,logpost)
        sums=np.bincount(index,weights=np.exp(logpost-maximum[index]),minlength=size)
        with np.errstate(divide='ignore'):return maximum+np.log(sums)

    def predict(self,y,hidden,identity=0,export_support_marginals=False):
        started=time.perf_counter();m=self.model
        y,hidden=validate_response(y,hidden,m.J)
        obs=np.flatnonzero(np.isfinite(y)&~hidden);targets=np.flatnonzero(np.isfinite(y)&hidden)
        if not len(targets): raise ValueError('No masked outcomes')
        c=np.where(np.isfinite(y),y-1,0).astype(int)
        lw=self.logprior.copy()
        for j in obs:self._add_emission(lw,j,c[j])
        normalizer=float(logsumexp(lw))
        if not np.isfinite(normalizer): raise ArithmeticError('Impossible conditioning event')
        lw-=normalizer
        w=np.exp(lw);lost=w==0
        loglost=float(logsumexp(lw[lost])) if lost.any() else -np.inf
        logcat=[];support_exports=[];fallback=0
        for j in targets:
            ix=self._full_index(j);size=1<<len(self.supports[j])
            mass=np.bincount(ix,weights=w,minlength=size)
            with np.errstate(divide='ignore'):lm=np.log(mass)
            cat=logsumexp(lm[:,None]+self.emissions[j],axis=0)
            # Each missing state's emission<=1. If its total posterior mass
            # could materially change a tiny category probability, recompute
            # every support mass using a group-specific logsumexp scale.
            if np.any(loglost-cat>math.log(2e-13)):
                lm=self._stable_support_logmass(ix,lw,size)
                cat=logsumexp(lm[:,None]+self.emissions[j],axis=0);fallback+=1
            logcat.append(cat)
            if export_support_marginals:support_exports.append(dict(item=int(j),support=self.supports[j].tolist(),logmass=lm))
        logcat=np.asarray(logcat)
        if np.max(abs(np.exp(logcat).sum(axis=1)-1))>2e-10:raise ArithmeticError('Predictive category normalization')
        # Full masked-vector integration uses the shared shallow state, so no
        # product-of-marginals substitution occurs here.
        joint=lw.copy()
        for j in targets:self._add_emission(joint,j,c[j])
        logjoint=float(logsumexp(joint))
        result=dict(targets=targets,log_probability=logcat[np.arange(len(targets)),c[targets]],
                    all_log_probabilities=logcat,log_joint=logjoint,method='sparse_support_lookup_EXACT_PROTOTYPE',
                    integration_diagnostic=0.,integration_pass=True,normal_CDFs_per_prediction=0,
                    model_threshold_identity=self.identity,observed_log_normalizer=normalizer,
                    underflowed_posterior_log_mass_bound=loglost,stable_log_bincount_fallbacks=fallback,
                    elapsed_seconds=time.perf_counter()-started,
                    scientific_scope='Exact fixed-parameter binary summation; no model-fit uncertainty.')
        if export_support_marginals:result['support_marginals']=support_exports
        return result
