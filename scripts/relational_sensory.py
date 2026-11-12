"""Fixed visible same-rank counts projected into anatomical sensory neurons."""
import hashlib
import numpy as np
from flylatro.fly.encoder import Stimulus

VERSION='visible-rank-companions-v1'


def companion_counts(obs):
    hand=obs['hand'];result=np.zeros(hand.shape[:2],dtype=np.float32)
    for row in range(len(hand)):
        n=int(obs['hand_len'][row]);cards=hand[row,:n]
        valid=(cards[:,37]==0)&(cards[:,23]==0)&(cards[:,:13].sum(axis=1)==1)
        ranks=cards[:,:13].argmax(axis=1)
        for i in range(n):
            if valid[i]:
                companions=int(np.sum(valid&(ranks==ranks[i])))-1
                result[row,i]=min(3,companions)
    return result


class RelationalEncoder:
    """Shared headroom preserves identity; relationship energy is exactly matched.

    On 30 reserved ALPNs, halve the original drive and add up to 75 Hz from
    companion count/3. No clipping occurs. Every slot owns three distinct roots.
    Wrong-slot controls cyclically shift counts among occupied slots after each
    display permutation; they preserve total additional sensory drive exactly.
    """
    def __init__(self,base,modes,replicates):
        self.base=base;self.modes=np.asarray(modes);self.replicates=np.asarray(replicates)
        self.mapping=base.mapping
        self.indices={int(rep):np.random.default_rng(42420+int(rep)).choice(
            self.mapping.available_alpn_indices,size=(10,3),replace=False)
            for rep in np.unique(self.replicates)}
        self.sha256=hashlib.sha256((VERSION+repr(self.indices)+repr(self.modes.tolist())).encode()).hexdigest()

    def encode(self,obs):
        old=self.base.encode(obs);rates=old.rates_hz.copy();counts=companion_counts(obs)
        if len(rates)!=len(self.modes):raise ValueError('declared lane layout differs')
        for row,(mode,rep) in enumerate(zip(self.modes,self.replicates)):
            indices=self.indices[int(rep)];rates[row,indices.ravel()]*=.5
            values=counts[row].copy();n=int(obs['hand_len'][row])
            if mode=='wrong':values[:n]=np.roll(values[:n],1)
            elif mode=='off':values[:]=0
            elif mode!='true':raise ValueError('unknown relationship mode')
            rates[row,indices.ravel()]+=np.repeat(values*25,3)
        if not np.isfinite(rates).all() or rates.max()>self.mapping.max_rate_hz+1e-5:
            raise ValueError('invalid relational stimulation')
        rates.setflags(write=False)
        return Stimulus(rates,VERSION,self.sha256)
