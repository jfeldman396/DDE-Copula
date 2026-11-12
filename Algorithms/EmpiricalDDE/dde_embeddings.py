"""Exact fixed-parameter DDE embeddings from personality responses only.

No fitting, classifiers, ideology/outcome inputs or data-file readers. Reuses
the sealed sparse-exact tables without editing their audited implementation.
Posterior means concatenate E[A1|Y] then E[A2|Y]. MAP maximizes the full joint
binary posterior, not a marginal shallow MAP followed by a top-layer choice.
"""
from pathlib import Path
from datetime import datetime, timezone
from itertools import product
import hashlib,json,math,os,sys,time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from scipy.special import logsumexp

HERE=Path(__file__).resolve().parent
from dde_predict import DDEModel,calibrate,states,validate_response
from sparse_exact_prototype import SparseExactPredictor,_hash_arrays

SEALED_SPARSE_EXACT_SHA256='e372401dbd99305bd84cfd6273bb2b8269100a14d9f5681ace661e40396fa366'


def _native_ids(value,width):
    if value is None:return np.arange(1,width+1,dtype=np.int64)
    a=np.asarray(value)
    if a.ndim!=1 or len(a)!=width or not np.isfinite(a.astype(float)).all():
        raise ValueError('Native IDs must be a finite layer-length vector')
    if not np.all(a.astype(float)==np.floor(a.astype(float))) or np.any(a.astype(float)<1):
        raise ValueError('Positive integer native IDs required')
    a=a.astype(np.int64)
    if len(np.unique(a))!=width:raise ValueError('Native IDs must be unique within a layer')
    return a


class DDEEmbedder:
    def __init__(self,model,thresholds,native_ids1=None,native_ids2=None,cache=None,
                 max_states=1<<22,low_bits=16,memory_budget_mb=768):
        source=HERE/'sparse_exact_prototype.py'
        if hashlib.sha256(source.read_bytes()).hexdigest()!=SEALED_SPARSE_EXACT_SHA256:
            raise ValueError('Sparse-exact source differs from independently audited snapshot')
        if cache is None:
            cache=SparseExactPredictor(model,thresholds,max_states=max_states,
                                      low_bits=low_bits,memory_budget_mb=memory_budget_mb)
        if not isinstance(cache,SparseExactPredictor):raise TypeError('Audited exact cache required')
        expected=_hash_arrays([np.asarray(model.B1,float),np.asarray(model.B2,float),
                               np.asarray(model.gamma,float).ravel(),np.asarray(model.prop,float).ravel(),
                               np.asarray(thresholds,float)])
        if expected!=cache.identity:raise ValueError('Embedding cache/model/cutpoint identity mismatch')
        self.cache=cache;self.model=cache.model;self.thresholds=cache.thresholds
        m=self.model;self.ids1=_native_ids(native_ids1,m.K1);self.ids2=_native_ids(native_ids2,m.K2)
        # Only the two bit halves are materialized, never S byK1 full states.
        self.alo=states(cache.low_bits);self.ahi=states(m.K1-cache.low_bits)
        self.top=states(m.K2)
        self.names=[f'A1[{x}]' for x in self.ids1]+[f'A2[{x}]' for x in self.ids2]
        self.metadata=dict(dimension=m.K1+m.K2,layer_widths=[m.K1,m.K2],
            native_ids1=self.ids1.tolist(),native_ids2=self.ids2.tolist(),feature_names=self.names,
            feature_order='All shallow A1 coordinates, followed by all upper A2 coordinates',
            native_ids_source='Explicit IDs where supplied; otherwise compact1based indices',
            model_threshold_identity=cache.identity,sparse_source_sha256=SEALED_SPARSE_EXACT_SHA256,
            mathematical_scope='Trained-two-layer DDE including a possible CSP reduction to zero retained upper factors',
            map_tie_rule='Smallest little-endian joint integer a_integer+(u_integer<<K1)',
            missing_policy='Every finite personality response conditions inference; NaNs give no likelihood term',
            outcome_inputs='None; this API accepts personality responses only',
            exact=True,parameter_uncertainty_included=False)

    def embed(self,y,identity=0,include_top_posterior=True):
        started=time.perf_counter();m=self.model;p=self.cache
        y,_=validate_response(y,np.zeros(m.J,dtype=bool),m.J)
        observed=np.flatnonzero(np.isfinite(y));cat=np.where(np.isfinite(y),y-1,0).astype(int)
        evidence=p.logprior.copy()
        for j in observed:p._add_emission(evidence,j,cat[j])
        lognormal=float(logsumexp(evidence))
        if not np.isfinite(lognormal):raise ArithmeticError('Zero/nonfinite personality likelihood')
        logpost=evidence-lognormal;w=np.exp(logpost)
        shallow_norm=float(w.sum())
        if abs(shallow_norm-1)>2e-10:raise ArithmeticError('Shallow posterior normalization')
        w/=shallow_norm
        block=w.reshape(p.nhi,p.nlo)
        e1=np.r_[block.sum(axis=0)@self.alo,block.sum(axis=1)@self.ahi]
        # Given A1, the observed Gaussian/ordinal likelihood has no further
        # A2 dependence. Compute each joint(A1,A2,Y) directly, preserving its
        # full hierarchical prior and the exact all-personality likelihood.
        likelihood=(evidence-p.logprior).reshape(p.nhi,p.nlo)
        prior_log_normalization=p.cache_diagnostic['prior_log_normalization_error']
        with np.errstate(divide='ignore'):
            topprior=np.where(self.top==1,np.log(m.prop),np.log1p(-m.prop)).sum(axis=1)
        logtop=np.empty(len(self.top));best=-np.inf;best_a=0;best_u=0;ties=0
        for u,upper in enumerate(self.top):
            if not np.isfinite(topprior[u]):logtop[u]=-np.inf;continue
            logits=m.B2[:,0]+m.B2[:,1:]@upper
            base=float(topprior[u]-np.logaddexp(0,logits).sum()-prior_log_normalization)
            low=self.alo@logits[:p.low_bits]+base
            high=self.ahi@logits[p.low_bits:]
            subtotal=-np.inf
            for h in range(p.nhi):
                joint=likelihood[h]+low+high[h]
                subtotal=np.logaddexp(subtotal,logsumexp(joint))
                i=int(np.argmax(joint));value=float(joint[i])
                if value>best:
                    best=value;best_a=i+h*p.nlo;best_u=u;ties=int(np.count_nonzero(joint==value))
                elif value==best:
                    ties+=int(np.count_nonzero(joint==value))
                    # Upper and high/low are traversed in the declared
                    # little-endian joint integer order, so keep first tie.
            logtop[u]=subtotal-lognormal
        top_log_norm=float(logsumexp(logtop))
        if abs(top_log_norm)>2e-10:raise ArithmeticError('Joint upper posterior normalization')
        topweights=np.exp(logtop-top_log_norm)
        e2=topweights@self.top
        means=np.r_[e1,e2]
        if not np.isfinite(means).all() or np.min(means)<-2e-12 or np.max(means)>1+2e-12:
            raise ArithmeticError('Invalid binary posterior expectation')
        clipped=np.clip(means,0,1)
        a=((best_a>>np.arange(m.K1,dtype=np.uint64))&1).astype(np.uint8)
        u=((best_u>>np.arange(m.K2,dtype=np.uint64))&1).astype(np.uint8)
        feature=np.r_[a,u]
        result=dict(posterior_mean=clipped,posterior_mean_a1=clipped[:m.K1],posterior_mean_a2=clipped[m.K1:],
            joint_map=feature,joint_map_a1=a,joint_map_a2=u,
            map_log_posterior=float(best-lognormal),map_posterior_probability=float(np.exp(best-lognormal)),
            joint_map_a1_integer=best_a,joint_map_a2_integer=best_u,
            joint_map_integer=best_a+(best_u<<m.K1),exact_MAP_tie_count=ties,
            observed_count=len(observed),missing_count=m.J-len(observed),identity=identity,
            log_personality_probability=lognormal,model_threshold_identity=p.identity,
            normalization=dict(shallow_probability_sum=shallow_norm,top_log_probability_sum=top_log_norm,
                               mean_roundoff_clipping=float(np.max(abs(clipped-means)))),
            integration_diagnostic=0.,integration_pass=True,method='exact_binary_DDE_personality_embedding',
            elapsed_seconds=time.perf_counter()-started,metadata=self.metadata)
        if include_top_posterior:
            result.update(top_states=self.top,top_posterior_probabilities=topweights,
                          top_log_posterior_probabilities=logtop-top_log_norm)
        return result

    def transform(self,Y,identities=None):
        Y=np.asarray(Y,float)
        if Y.ndim!=2 or Y.shape[1]!=self.model.J:raise ValueError('Expected personality matrix rows byJ')
        if identities is None:identities=np.arange(len(Y))
        if len(identities)!=len(Y):raise ValueError('One identity per respondent')
        rows=[self.embed(y,identity=i,include_top_posterior=False) for y,i in zip(Y,identities)]
        width=self.model.K1+self.model.K2
        return dict(posterior_mean=np.asarray([r['posterior_mean'] for r in rows]).reshape(len(Y),width),
                    joint_map=np.asarray([r['joint_map'] for r in rows],dtype=np.uint8).reshape(len(Y),width),
                    row_diagnostics=[{k:r[k] for k in ['identity','observed_count','missing_count','normalization',
                                                       'map_log_posterior','exact_MAP_tie_count','elapsed_seconds']} for r in rows],
                    metadata=self.metadata)


def _reference_joint(model,thresholds,y):
    """Independent100-digit joint table; no sparse-cache/prior helper imports."""
    import mpmath as mp
    from reference_bruteforce import interval,number
    m=model
    with mp.workdps(100):
        table=[]
        for a in product((0,1),repeat=m.K1):
            for u in product((0,1),repeat=m.K2):
                prior=mp.mpf(1)
                for bit,q in zip(u,m.prop):prior*=number(q) if bit else 1-number(q)
                for k,bit in enumerate(a):
                    eta=number(m.B2[k,0])+sum(number(b)*x for b,x in zip(m.B2[k,1:],u))
                    prob=1/(1+mp.exp(-eta));prior*=prob if bit else 1-prob
                weight=prior
                for j,value in enumerate(y):
                    if math.isnan(float(value)):continue
                    category=int(value)-1;mean=number(m.B1[j,0])+sum(number(b)*x for b,x in zip(m.B1[j,1:],a))
                    sd=mp.sqrt(number(m.gamma[j]))
                    weight*=interval((number(thresholds[j,category])-mean)/sd,
                                     (number(thresholds[j,category+1])-mean)/sd)
                key=sum(bit<<k for k,bit in enumerate(a))+sum(bit<<(m.K1+k) for k,bit in enumerate(u))
                table.append((a,u,weight,key))
        z=mp.fsum(x[2] for x in table)
        mean=[float(mp.fsum(w*list(a+u)[k] for a,u,w,key in table)/z) for k in range(m.K1+m.K2)]
        # Deterministic tie selection independently sorts the actual joint keys.
        best=sorted(table,key=lambda x:(-x[2],x[3]))[0]
        return dict(mean=np.array(mean),map=np.array(best[0]+best[1]),mapkey=best[3],
                    lognormal=float(mp.log(z)),map_log_probability=float(mp.log(best[2]/z)),
                    table=[dict(a=a,u=u,p=float(w/z),key=key) for a,u,w,key in table])
