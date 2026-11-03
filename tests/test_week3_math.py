"""Executable counterexamples for Week 3 universal claims, not CNN certificates."""
from fractions import Fraction
import unittest
import numpy as np


class Week3Counterexamples(unittest.TestCase):
    def test_h02_overlap_does_not_bound_ordinary_distance(self):
        recording=np.tile(np.arange(192,dtype=np.float32),(6,1))
        old,new=recording[:,:128],recording[:,64:]
        np.testing.assert_array_equal(old[:,64:],new[:,:64])
        np.testing.assert_array_equal(new-old,np.full((6,128),64,np.float32))
        self.assertGreater(np.linalg.norm(new-old),1700)
        # Scaling a recording preserves overlap but makes the distance unbounded.
        distance=(new-old).astype(np.float64)
        self.assertAlmostEqual(np.linalg.norm(10*distance)/np.linalg.norm(distance),10)

    def test_h07_direct_margin_ibp_is_incomplete(self):
        # h1=h2=ReLU(t+1). Identical dependencies disappear in independent intervals.
        t=np.arange(-512,513,dtype=np.float64)/512  # exactly represented dyadics
        h1=np.maximum(0,t+1);h2=np.maximum(0,t+1)
        np.testing.assert_array_equal(1+h1-h2,np.ones_like(t))
        hidden_lo=np.array([0.,0.]);hidden_hi=np.array([2.,2.])
        margin_weights=np.array([1.,-1.])
        ibp_lower=1+np.maximum(margin_weights,0)@hidden_lo+np.minimum(margin_weights,0)@hidden_hi
        self.assertEqual(ibp_lower,-1.)

    def test_h15_recentered_hit_can_cross_decision_boundary(self):
        logits=lambda x:np.array([1-x,0.])
        anchor=0.;radius=.6;hit=.6;later=1.2
        self.assertGreater(1-(anchor+radius),0)  # whole original interval has class0
        self.assertLessEqual(abs(hit-anchor),radius)
        self.assertLessEqual(abs(later-hit),radius)  # unsafe recenter would accept
        self.assertGreater(abs(later-anchor),radius)
        self.assertEqual(logits(anchor).argmax(),0)
        self.assertEqual(logits(later).argmax(),1)

    def test_h19_sound_singleton_has_no_future_coverage(self):
        domain=lambda x:x==0
        self.assertTrue(domain(0))
        future=np.arange(1,11)
        self.assertEqual(sum(domain(x) for x in future),0)

    def test_h20_nonzero_reuse_can_still_cost_more(self):
        n=100;reused=50;tf=1.;tb=10.;builds=20;per_query_gate=.1
        baseline=n*tf
        ecdr=(n-reused)*tf+builds*tb+n*per_query_gate
        self.assertGreater(reused,0)
        self.assertGreater(ecdr,baseline)

    def test_h22_real_positive_margin_can_round_to_wrong_tie(self):
        exact=[Fraction(1),Fraction(1)+Fraction(1,2**25)]
        self.assertGreater(exact[1]-exact[0],0)
        executable=np.array([float(v) for v in exact],dtype=np.float32)
        self.assertEqual(executable[0],executable[1])
        self.assertEqual(executable.argmax(),0)  # real winner is1


if __name__=='__main__':unittest.main()
