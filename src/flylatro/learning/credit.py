"""Experimental executed-choice eligibility, without learned decoder parameters."""
from __future__ import annotations

import hashlib
import numpy as np

from flylatro.fly.mushroom_body.plasticity import CHOICE_GATED_PLASTICITY_VERSION


class SelectedMotorCredit:
    """Map selected fixed motor pools to anatomical KC→MBON target edges.

    Only a compact pool/post mask crosses to CUDA. Edge masks and index tensors
    are reused there. No reward, action-quality oracle or fitted policy is used.
    """

    def __init__(self, motor, plasticity):
        if motor.mapping.mode != "mbon_direct":
            raise ValueError("choice-gated plasticity requires mbon_direct motor output")
        self.motor = motor
        self.plasticity = plasticity
        posts, self.edge_posts = np.unique(plasticity.topology.post_root_ids, return_inverse=True)
        lookup = {int(root): i for i, root in enumerate(posts)}
        self.pool_posts = np.zeros((motor.mapping.routing.pool_count, len(posts)), dtype=np.bool_)
        for pool, indices in enumerate(motor.mapping.pool_indices):
            for index in indices:
                root = int(motor.mapping.output_root_ids[index])
                if root in lookup:
                    self.pool_posts[pool, lookup[root]] = True
            if not self.pool_posts[pool].any():
                raise ValueError(f"motor pool {pool} has no anatomical plastic KC→MBON inputs")
        self.edge_counts = np.bincount(self.edge_posts, minlength=len(posts))
        self.card_pool_ids = np.asarray(motor.mapping.routing.card_pool_ids, dtype=int)
        self.card_posts = self.pool_posts[self.card_pool_ids].any(axis=0)
        self._device_indices = None
        self._post_credit = None
        self._edge_credit = None
        self.metrics = {}
        digest = hashlib.sha256()
        digest.update(posts.astype('<i8').tobytes())
        digest.update(self.edge_posts.astype('<i8').tobytes())
        digest.update(self.pool_posts.tobytes())
        self.manifest = {
            'version': CHOICE_GATED_PLASTICITY_VERSION,
            'assignment_sha256': digest.hexdigest(),
            'motor_mapping_sha256': motor.mapping.structure_sha256,
            'policy': 'union of selected competing-head motor pools; forced and inactive heads excluded',
            'eligibility': 'clip(decay * previous + normalized_coactivity * selected_post_gate)',
            'delayed_credit': 'previous selected traces decay normally; terminal update precedes trace clearing',
            'evidence': 'engineering action-feedback hypothesis; not an anatomical DAN compartment assignment',
            'external_trainable_parameter_count': 0,
        }

    def mask(self, actions, masks):
        pools = self.motor.selected_pool_mask(actions, masks)
        # A logical union avoids counting shared pools/neurons more than once.
        posts = np.any(pools[:, :, None] & self.pool_posts[None, :, :], axis=1)
        counts = posts @ self.edge_counts
        intended_card_posts = np.any(pools[:, self.card_pool_ids, None]
            & self.pool_posts[None, self.card_pool_ids, :], axis=1)
        admitted_card = posts & self.card_posts[None, :]
        aliased_card = admitted_card & ~intended_card_posts
        card_counts = admitted_card @ self.edge_counts
        alias_counts = aliased_card @ self.edge_counts
        self.metrics = {
            'credit/selected_pool_count_mean': float(pools.sum(axis=1).mean()),
            'credit/new_eligible_edge_fraction_mean': float(counts.mean() / len(self.edge_posts)),
            'credit/forced_only_learner_fraction': float(np.mean(~pools.any(axis=1))),
            'credit/card_alias_edge_fraction_mean': float(np.mean(
                np.divide(alias_counts, card_counts, out=np.zeros_like(alias_counts, dtype=float), where=card_counts > 0))),
            'credit/card_alias_trial_fraction': float(np.mean(aliased_card.any(axis=1))),
        }
        if not self.plasticity.is_torch:
            return posts[:, self.edge_posts]
        import torch
        device = self.plasticity.state.device
        if self._device_indices is None:
            self._device_indices = torch.tensor(self.edge_posts, dtype=torch.long, device=device)
        if self._post_credit is None or tuple(self._post_credit.shape) != posts.shape:
            self._post_credit = torch.empty(posts.shape, dtype=torch.bool, device=device)
            self._edge_credit = torch.empty((len(posts), len(self.edge_posts)), dtype=torch.bool, device=device)
        self._post_credit.copy_(torch.from_numpy(posts))
        torch.index_select(self._post_credit, 1, self._device_indices, out=self._edge_credit)
        return self._edge_credit
