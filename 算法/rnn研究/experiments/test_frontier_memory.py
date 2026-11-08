"""Structural checks for the frontier benchmark."""
import unittest

import torch
from torch.nn import functional as F

from conditional_readout import ConditionalReadout
from frontier_data import TASKS, episode, oracle
from frontier_models import NAMES, build
from shared_address_graph import SharedAddressGraph


class FrontierMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_oracle_and_counterfactuals(self):
        for task in TASKS:
            difficulty=3 if task in ('pointer','arithmetic') else 1
            r=episode(task,'scattered',2048,difficulty,f'test/{task}')
            x=r['variants'][0]
            self.assertEqual([oracle(v) for v in r['variants']],r['targets'])
            self.assertEqual(x[:-1],r['variants'][1][:-1])
            self.assertNotEqual(r['targets'][0],r['targets'][1])
            for v in r['variants'][2:]:
                self.assertEqual(sum(a!=b for a,b in zip(x,v,strict=True)),1)
            if task=='latest': self.assertEqual(r['targets'][0],r['targets'][-1])

    def test_streaming_chunks_match_and_capacity_is_fixed(self):
        model=build('adaptive').eval()
        x=torch.tensor([episode('pointer','scattered',2048,3,'test/stream')['variants'][0]])
        state=model.initial(len(x),x.device)
        with torch.no_grad():
            for token in x[:,:-2].split(1,dim=1):
                state=model.push(state,token)
                self.assertEqual(tuple(state.slots.shape),(1,32,32))
                self.assertLess(state.pending.shape[1],64)
            torch.testing.assert_close(model.answer(state,x[:,-2:]),model(x))

    def test_active_edges_and_training_gradient(self):
        model=build('adaptive')
        r=episode('arithmetic','scattered',128,3,'test/edges')
        x=torch.tensor(r['variants'])
        model.eval()
        with torch.no_grad():
            logits,edges=model(x,return_edges=True)
            self.assertEqual(sum(e[0].numel() for e in edges),64)
            torch.testing.assert_close(model(x,forced=edges),logits)
        model.train();F.cross_entropy(model(x),torch.tensor(r['targets'])).backward()
        for param in (model.write_q.weight,model.write_k.weight,model.read.q.weight,model.read.k.weight):
            self.assertTrue(param.grad.isfinite().all())
            self.assertGreater(param.grad.abs().sum().item(),0)

    def test_parameter_budget_and_tcn_field(self):
        for name in NAMES:
            model=build(name)
            self.assertTrue(90_000<=sum(p.numel() for p in model.parameters())<=110_000)
            if name=='TCN': self.assertEqual(1+sum(l.dilation[0] for l in model.core),4096)

    def test_shared_address_changes_destination_and_supports_edge_reuse(self):
        row = episode('lookup', 'scattered', 384, 1, 'test/shared-address')
        x = torch.tensor(row['variants'])
        model = SharedAddressGraph()
        model.eval()
        with torch.no_grad():
            logits, edges = model(x, return_edges=True)
            self.assertEqual(tuple(edges[0].shape), (3, 4, 4))
            for i in (0, 1):
                destination = row['variants'][i].index(row['variants'][i][-1]) + 1
                self.assertEqual(edges[0][i,0,0].item(), destination)
            torch.testing.assert_close(model(x, forced=edges), logits)
            _, frozen = model(x[1:2], forced=[edges[0][:1]], return_edges=True)
            torch.testing.assert_close(frozen[0], edges[0][:1])
        model.train()
        torch.testing.assert_close(model(x), logits)
        F.cross_entropy(model(x), torch.tensor(row['targets'])).backward()
        self.assertTrue(model.address.weight.grad.isfinite().all())
        self.assertGreater(model.address.weight.grad.abs().sum().item(), 0)

    def test_conditional_readout_can_change_instruction_with_fixed_evidence(self):
        row = episode('arithmetic', 'scattered', 384, 2, 'test/conditional-readout')
        x = torch.tensor(row['variants'][:2])
        model = ConditionalReadout('conditioned', neutral_routing=True)
        model.eval()
        with torch.no_grad():
            query, history, edges = model.composition_inputs(x)
            torch.testing.assert_close(history[0], history[1])
            self.assertTrue(torch.equal(edges[0], edges[1]))
            self.assertFalse(torch.equal(query[0], query[1]))
            changed = model(x, readout_query=x.flip(0)[:, -2:])
            torch.testing.assert_close(changed, model(x).flip(0))
            original = SharedAddressGraph()
            baseline = ConditionalReadout('shared')
            baseline.load_state_dict(original.state_dict())
            torch.testing.assert_close(original.eval()(x), baseline.eval()(x))

    def test_relation_bank_and_graph_can_be_separated(self):
        row = episode('arithmetic', 'scattered', 64, 2, 'test/relation-bank')
        x = torch.tensor(row['variants'][:2])
        model = ConditionalReadout('conditioned', neutral_routing=True, binding=True, relation_bank=True)
        model.eval()
        with torch.no_grad():
            _, combined, edges = model.composition_inputs(x)
            _, relations, _ = model.composition_inputs(x, evidence='relations')
            _, graph, _ = model.composition_inputs(x, evidence='graph')
            torch.testing.assert_close(combined, relations+graph)
            torch.testing.assert_close(relations[0], relations[1])
            self.assertTrue(torch.equal(edges[0], edges[1]))
            self.assertEqual(sum(p.numel() for p in model.parameters()),
                             sum(p.numel() for p in SharedAddressGraph().parameters()))


if __name__=='__main__': unittest.main()
