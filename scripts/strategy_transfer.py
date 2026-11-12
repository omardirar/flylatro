"""Exploratory transfer to raw discard membership and the fly's own PLAY."""
import numpy as np

from flylatro.analysis.card_selection import plastic_hashes
from flylatro.env.upstream_contract import empty_action_batch
from flylatro.seeds import derive_seed
from discard_strategy import VERSION
from strategy_puzzles import resample_future
from train_discard_play_curriculum import permute_rows, translate


def fly_transfer(study, *, draws=8):
    before=plastic_hashes(study.local.state)
    result={}
    for label,initial in [('initial',True),('trained',False)]:
        rows=[]
        efficacy=study.local.state.initial_efficacy if initial else study.local.state.efficacy
        for index in study.corpus.indices('fresh'):
            selected,_,_,neural,orders,legal,raw=study.choose([index]*study.n,
                ('fly-transfer',index),initial=initial,free_cards=True)
            forced={k:v.copy() for k,v in legal.items()}
            forced['action_type_mask'][:]=False;forced['action_type_mask'][:,0]=True
            own=study.agent.motor.decode(neural.output_activity,forced,deterministic=True)
            own_native=translate(own,orders);discard_native=translate(selected,orders)
            discarded=selected['action_type']==1
            for draw in range(draws):
                for lane in range(study.n):
                    study.corpus.restore(study.env,lane,index)
                    resample_future(study.env,lane,derive_seed(VERSION,index,draw,'fly-transfer-future'))
                executed={k:v.copy() for k,v in own_native.items()}
                for k in executed:executed[k][discarded]=discard_native[k][discarded]
                first=study.env.step(executed)
                infos=[dict(x) if not discarded[i] else None for i,x in enumerate(first.infos)]
                plays=[executed['cards'][i,:int(executed['n_cards'][i])].tolist() if not discarded[i] else None
                       for i in range(study.n)]
                if discarded.any():
                    after,masks=study.env.observe()
                    for k in after:after[k][~discarded]=raw[k][~discarded]
                    _,initial_masks=study.env.observe()
                    # Restore valid original legal masks for inactive rows;
                    # those rows are processed but never execute these PLAYs.
                    for k in masks:masks[k][~discarded]=legal[k][~discarded]
                    masks['action_type_mask'][:]=False;masks['action_type_mask'][:,0]=True
                    next_orders=[np.r_[np.random.default_rng(derive_seed(VERSION,index,draw,int(rep),'fly-post-display')).permutation(8),8,9]
                                 for rep in study.ids]
                    visible,play_masks=permute_rows(after,masks,next_orders)
                    activity=study.agent.processor.process(visible,
                        fly_seeds=[derive_seed(VERSION,index,draw,int(rep),'fly-post-neural') for rep in study.ids],efficacy=efficacy)
                    play=translate(study.agent.motor.decode(activity.output_activity,play_masks,deterministic=True),next_orders)
                    follow=empty_action_batch(study.n);follow['n_cards'][:]=0
                    _,native_masks=study.env.observe()
                    for lane in range(study.n):
                        if discarded[lane]:
                            for k in follow:follow[k][lane]=play[k][lane]
                            plays[lane]=play['cards'][lane,:int(play['n_cards'][lane])].tolist()
                        else:
                            kind=int(np.flatnonzero(native_masks['action_type_mask'][lane])[0])
                            if kind in (0,1):raise ValueError('invalid inactive transfer phase')
                            follow['action_type'][lane]=kind
                    second=study.env.step(follow)
                    for lane in np.flatnonzero(discarded):infos[lane]=dict(second.infos[lane])
                rows.append({'case_index':index,'draw':draw,'discarded':discarded.tolist(),
                    'discard_sets_native':[discard_native['cards'][i,:4].tolist() if discarded[i] else None for i in range(study.n)],
                    'play_cards_native':plays,'cleared':[bool(i['reward_components'].get('blind_clear',0)>0) for i in infos]})
        values=np.asarray([r['cleared'] for r in rows],dtype=float)
        summary={name:{'per_learner':values[:,study.roles==role].mean(axis=0).tolist(),
            'mean':float(values[:,study.roles==role].mean())} for role,name in enumerate(('original','normalised','frozen'))}
        result[label]={'rows':rows,'summary':summary}
    if plastic_hashes(study.local.state)!=before:raise ValueError('fly transfer changed state')
    result['scope']='raw discard membership, fixed discard count four, fly chooses PLAY count/cards; last-hand puzzle transfer, not full games'
    result['draws']=draws;result['learning_disabled']=True
    return result
