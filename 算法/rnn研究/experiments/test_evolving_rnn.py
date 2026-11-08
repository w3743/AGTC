import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from edge_adamw import EdgeAdamW
from evolving_rnn import EvolvingRNN, Survival
from evolving_rnn_eval import questions
from evolving_rnn_acquire import paired
from evolving_rnn_lookup import capacity_batch, lookup_queries, loss_for, pack_records, query_ticks
from evolving_rnn_sleep_guard import SleepGuard, TrainingCalibration
from evolving_rnn_two_cues import cue_groups
from evolving_rnn_transfer import guarded_update
from immw import Factors, ask, create_world, suffix_ceiling
from immw_report import reference


class EvolvingRNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_full_start_matches_dense_and_probe_reaches_dormant_edge(self):
        torch.manual_seed(2)
        dense = EvolvingRNN(24,'dense')
        torch.manual_seed(2)
        model = EvolvingRNN(24,'evolve',Survival(traces=(2,16,64)))
        x,y = questions('lookup',4,4,torch.Generator().manual_seed(8))
        x,y = x.flatten(0,1),y.flatten()
        torch.testing.assert_close(dense(x),model(x))
        model.active[0,0] = False
        loss = F.cross_entropy(model(x,probe=True),y)
        loss.backward()
        self.assertEqual(model.core.weight_hh_l0.grad[0,0].item(),0.)
        self.assertNotEqual(model.probe.grad[0,0].item(),0.)
        torch.testing.assert_close(model.temporal_gradients((float('inf'),))[0],model.probe.grad,
                                   atol=1e-7,rtol=1e-4)

    def test_temporal_gradient_includes_supervision_before_final_query(self):
        torch.manual_seed(2)
        model = EvolvingRNN(24,'evolve',Survival(traces=(2,16,64)))
        x,y = lookup_queries(16,2,torch.Generator().manual_seed(21))
        auxiliary = torch.nn.Linear(24,16)
        loss = loss_for(model,auxiliary,'retention-readout',True,x,y,probe=True)
        loss.backward()
        torch.testing.assert_close(model.temporal_gradients((float('inf'),))[0],model.probe.grad,
                                   atol=1e-7,rtol=1e-4)

    def test_explicit_sleep_and_regrow_updates_budget_without_target_count(self):
        torch.manual_seed(2)
        model = EvolvingRNN(16,'evolve',Survival(dormant_credit=1.30))
        optimizer = torch.optim.AdamW(model.parameters())
        model.active[0,0] = False
        model.life[0,0] = model.config.wake-.005
        model.life[0,1] = .005
        x = torch.tensor([[6,2,7],[6,1,7]])
        F.cross_entropy(model(x,probe=True),torch.tensor([2,1])).backward()
        model.probe.grad.zero_()
        model.probe.grad[0,0] = 1.
        optimizer.step()
        event = model.evolve(optimizer)
        self.assertTrue(model.active[0,0])
        self.assertFalse(model.active[0,1])
        self.assertEqual((event['slept'],event['woke']),(1,1))
        self.assertEqual(model.core.weight_hh_l0[0,1].item(),0.)

    def test_renewed_edge_matches_fresh_adamw_without_changing_old_edge(self):
        weight = torch.nn.Parameter(torch.tensor([.01,.02],dtype=torch.float64))
        baseline = torch.nn.Parameter(weight.detach().clone())
        fresh = torch.nn.Parameter(weight.detach()[0:1].clone())
        optimizer = EdgeAdamW([weight],weight,lr=.001)
        stock = torch.optim.AdamW([baseline],lr=.001)
        fresh_optimizer = torch.optim.AdamW([fresh],lr=.001)
        for opt,p in ((optimizer,weight),(stock,baseline)):
            p.grad = torch.zeros_like(p)
            opt.step()
            opt.state[p]['step'].fill_(1000)
        with torch.no_grad():
            fresh.copy_(weight[:1])
        optimizer.reset_edges(torch.tensor([False,False]),torch.tensor([True,False]))
        for i in range(3):
            g = (-1.)**i
            weight.grad = torch.full_like(weight,g)
            baseline.grad = weight.grad.clone()
            fresh.grad = weight.grad[:1].clone()
            optimizer.step()
            stock.step()
            fresh_optimizer.step()
            torch.testing.assert_close(weight[:1],fresh,atol=1e-14,rtol=1e-12)
            torch.testing.assert_close(weight[1:],baseline[1:],atol=0,rtol=0)

    def test_rejected_structural_update_restores_weights_and_optimizer(self):
        torch.manual_seed(2)
        model = EvolvingRNN(16,'evolve',Survival(dormant_credit=1.30))
        weight = model.core.weight_hh_l0
        optimizer = EdgeAdamW(model.parameters(),weight,lr=.001)
        model.active[0,0] = False
        model.life[0,0] = model.config.wake-.005
        model.life[0,1] = .005
        x = torch.tensor([[6,2,7],[6,1,7]])
        F.cross_entropy(model(x,probe=True),torch.tensor([2,1])).backward()
        model.probe.grad.zero_()
        model.probe.grad[0,0] = 1.
        optimizer.step()
        state = optimizer.state[weight]
        tensors = (weight,model.active,model.life,state['exp_avg'],state['exp_avg_sq'])
        original = [t.detach().clone() for t in tensors]
        with patch('evolving_rnn_transfer.calibration_metrics',
                   side_effect=[[(4,1.,.01)],[(4,.9,.2)]]):
            outcome = guarded_update(model,optimizer,[])
        self.assertFalse(outcome['accepted'])
        self.assertEqual((outcome['proposed_sleep'],outcome['proposed_wake']),(1,1))
        self.assertNotIn('edge_age',state)
        for got,expected in zip(tensors,original,strict=True):
            torch.testing.assert_close(got,expected,atol=0,rtol=0)

    def test_immw_counterfactual_and_observation_cutoff(self):
        f = Factors(2048,64,8)
        world = create_world(f,'test/interruptions')
        q = ask(world,15)
        self.assertEqual(world.interruptions,512)
        self.assertGreaterEqual(q['ages'][0],2048)
        self.assertLess(q['ages'][2],32)
        self.assertIn('interruption',q['edit_kinds'])
        self.assertEqual(tuple(map(reference,q['variants'])),q['targets'])
        self.assertFalse(suffix_ceiling(q,256))
        self.assertTrue(suffix_ceiling(q,len(q['variants'][0])))

    def test_shared_history_queries_and_source_edits(self):
        x,y = lookup_queries(4,12,torch.Generator().manual_seed(7),rule=1)
        x = x.reshape(12,2,4,-1)
        y = y.reshape(12,2,4)
        self.assertTrue(torch.equal(x[...,:-1],x[:,:,:1,:-1].expand_as(x[...,:-1])))
        self.assertTrue(torch.all((x[:,0]!=x[:,1]).sum(-1)==1))
        self.assertTrue(torch.all((y[:,0]!=y[:,1]).sum(-1)==1))
        for variants,labels in zip(x.tolist(),y.tolist(),strict=True):
            for same_history,answers in zip(variants,labels,strict=True):
                for tokens,answer in zip(same_history,answers,strict=True):
                    records = dict(zip(tokens[1:9:2],tokens[2:9:2],strict=True))
                    self.assertEqual(answer,records[8+(tokens[-1]-8+1)%4])
        self.assertTrue(torch.equal(query_ticks(x)[...,-4:],x[...,-1:].expand(*x.shape[:-1],4)))
        packed = pack_records(x)
        self.assertTrue(torch.equal((packed[...,1:5]-12)//4+8,x[...,1:9:2]))
        self.assertTrue(torch.equal((packed[...,1:5]-12)%4,x[...,2:9:2]))
        self.assertTrue(torch.equal(packed[...,5:],x[...,9:]))

    def test_dense_retention_losses_leave_the_original_readout_trainable(self):
        x,y = lookup_queries(32,2,torch.Generator().manual_seed(8))
        for method,vocab in (('retention-readout',12),('packed-retention',28)):
            model = EvolvingRNN(16,'dense',vocab=vocab)
            auxiliary = torch.nn.Linear(16,16)
            loss = loss_for(model,auxiliary,method,True,x,y)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(float(model.head.weight.grad.abs().sum()),0.)
            self.assertGreater(float(auxiliary.weight.grad.abs().sum()),0.)

    def test_new_memory_labels_depend_on_the_requested_cue_only(self):
        rng = torch.Generator().manual_seed(8)
        for channel in (0,1):
            x,y = paired(64,12,rng,channel)
            self.assertTrue(torch.all((x[:,0,None]!=x[:,1:]).sum(-1)==1))
            self.assertTrue(torch.all(y[:,0]!=y[:,1]))
            self.assertTrue(torch.equal(y[:,0],y[:,2]))
            self.assertTrue(torch.equal(y,x[:,:,1+2*channel]))

    def test_two_cue_groups_use_one_source_edit_and_two_queries(self):
        x,y = cue_groups(16,16,torch.Generator().manual_seed(9))
        self.assertEqual(tuple(y.shape),(16,3,2))
        self.assertTrue(torch.equal(x[:,:,0,:-1],x[:,:,1,:-1]))
        self.assertTrue(torch.all((x[:,0,None]!=x[:,1:]).sum(-1)==1))
        self.assertTrue(torch.all(y[:,0,0]!=y[:,1,0]))
        self.assertTrue(torch.equal(y[:,0,1],y[:,1,1]))
        self.assertTrue(torch.equal(y[:,0,0],y[:,2,0]))
        self.assertTrue(torch.all(y[:,0,1]!=y[:,2,1]))

    def test_capacity_curriculum_matches_all_keys_at_each_record_count(self):
        rng = torch.Generator().manual_seed(12)
        for step,expected in ((120,2),(900,3),(1600,4),(2812,4),(3881,4)):
            lookup,x,y,count = capacity_batch(2,step,rng)
            self.assertTrue(lookup)
            self.assertEqual(count,expected)
            self.assertEqual(len(x),24)
            for sequence,target in zip(x.tolist(),y.tolist(),strict=True):
                records = dict(zip(sequence[1:2*count+1:2],sequence[2:2*count+1:2],strict=True))
                self.assertEqual(len(records),count)
                self.assertEqual(records[sequence[-1]],target)
            model = EvolvingRNN(16,'dense')
            auxiliary = torch.nn.Linear(16,16)
            loss = loss_for(model,auxiliary,'capacity-curriculum',True,x,y,record_count=count)
            loss.backward()
            self.assertGreater(float(auxiliary.weight.grad.abs().sum()),0.)

    def test_fixed_learned_mask_reinitializes_weights_at_matched_density(self):
        mask = torch.eye(16,dtype=torch.bool)
        torch.manual_seed(2)
        model = EvolvingRNN(16,'fixed',fixed_edges=16,fixed_mask=mask)
        torch.manual_seed(2)
        dense = EvolvingRNN(16,'dense')
        self.assertTrue(torch.equal(model.active,mask))
        torch.testing.assert_close(model.core.weight_hh_l0,dense.core.weight_hh_l0*mask*4)

    def test_binary_conductance_keeps_live_weights_until_sleep(self):
        torch.manual_seed(2)
        gradual = EvolvingRNN(16,'evolve')
        binary = EvolvingRNN(16,'evolve',Survival(conductance_floor=1.))
        binary.load_state_dict(gradual.state_dict())
        for model in (gradual,binary):
            model.life.fill_(.1)
            model.active[0,0] = False
        reference = binary.core.weight_hh_l0*binary.active
        torch.testing.assert_close(binary.effective_recurrent(),reference)
        torch.testing.assert_close(gradual.effective_recurrent(),.6*reference)

    def test_paced_sleep_completes_the_same_candidate_batch(self):
        torch.manual_seed(2)
        config = Survival(conductance_floor=1.)
        x = torch.tensor([[6,2,7],[6,1,7]])
        y = torch.tensor([2,1])
        immediate, paced = (EvolvingRNN(16,'evolve',config) for _ in range(2))
        paced.load_state_dict(immediate.state_dict())
        optimizers = [torch.optim.AdamW(model.parameters()) for model in (immediate,paced)]
        for model,optimizer in zip((immediate,paced),optimizers,strict=True):
            model.life[0,:8] = .005
            F.cross_entropy(model(x,probe=True),y).backward()
            model.probe.grad.zero_()
            optimizer.step()
        expected = immediate.evolve(optimizers[0])
        queue = []
        partial = paced.evolve(optimizers[1],queue,sleep_window=4)
        self.assertEqual((expected['slept'],partial['eligible'],partial['slept']),(8,8,2))
        self.assertTrue((paced.life[0,:8][paced.active[0,:8]]<0).all())
        torch.testing.assert_close(paced.effective_recurrent(),paced.core.weight_hh_l0*paced.active)
        while queue:
            paced.finish_sleep(optimizers[1],queue.pop())
        for a,b in zip(immediate.state_dict().values(),paced.state_dict().values(),strict=True):
            torch.testing.assert_close(a,b)

    def test_sleep_guard_protects_one_group_without_resetting_its_life(self):
        torch.manual_seed(2)
        model = EvolvingRNN(16,'evolve',Survival(conductance_floor=1.))
        optimizer = torch.optim.AdamW(model.parameters())
        x = torch.tensor([[6,2,7],[6,1,7]])
        model.life[0,:8] = .005
        F.cross_entropy(model(x,probe=True),torch.tensor([2,1])).backward()
        model.probe.grad.zero_()
        optimizer.step()
        guard = SleepGuard(model,2,20,0)

        def measured(model,data,cut=None):
            damage = .1 if cut is not None and cut[0,0] else 0.
            return dict(loss=[damage,0.],joint=[1.,1.])

        with patch('evolving_rnn_sleep_guard.scores',side_effect=measured):
            event = model.evolve(optimizer,sleep_filter=guard)
        self.assertEqual((event['eligible'],event['slept']),(8,7))
        self.assertTrue(model.active[0,0])
        self.assertLess(model.life[0,0].item(),0.)
        self.assertTrue((model.life[0,1:8]==-model.config.cooldown).all())
        self.assertEqual(guard.log['checks'],9)

    def test_prefix_guard_keeps_the_survival_order(self):
        model = EvolvingRNN(16,'evolve',Survival(conductance_floor=1.))
        order = torch.tensor([4,5,6,7,0,1,2,3])
        model.life[0,order] = torch.arange(8)*.01-.1
        eligible = torch.zeros_like(model.active)
        eligible[0,:8] = True
        guard = SleepGuard(model,2,20,0,prefix=True)

        def measured(model,data,cut=None):
            damage = .1 if cut is not None and int(cut.sum())>2 else 0.
            return dict(loss=[0.,damage],joint=[1.,1.])

        with patch('evolving_rnn_sleep_guard.scores',side_effect=measured):
            chosen = guard(eligible)
        self.assertEqual(torch.where(chosen[0])[0].tolist(),[4,5])
        self.assertEqual(guard.log['checks'],3)

    def test_training_calibration_reuses_latest_batches_without_changing_examples(self):
        cache = TrainingCalibration()
        rng = torch.Generator().manual_seed(245289)
        last = {}
        for step in range(4401,4441):
            lookup,x,y,_ = capacity_batch(2,step,rng)
            cache.add(lookup,x,y)
            last[lookup,x.shape[-1]] = (x,y)
        for lookup,batches in zip((False,True),cache.data(),strict=True):
            for x,y in batches:
                actual_x,actual_y = last[lookup,x.shape[-1]]
                torch.testing.assert_close(x.flatten(0,1),actual_x)
                torch.testing.assert_close(y.flatten(),actual_y)
        thin = TrainingCalibration(groups_per_cell=1)
        thin.batches = cache.batches
        for selected,full in zip(thin.data(rotation=1),cache.data(),strict=True):
            for (x,y),(all_x,all_y) in zip(selected,full,strict=True):
                torch.testing.assert_close(x,all_x[1:2])
                torch.testing.assert_close(y,all_y[1:2])


if __name__=='__main__':
    unittest.main()
