import sys, unittest, io
sys.path.insert(0, '/home/pro/Downloads/research'); sys.path.insert(0, '/home/pro/Downloads/research/tests')
import numpy as np
import test_week4_membership as T
from ecdr.week4 import domains, reference

def run():
    suite = unittest.defaultTestLoader.loadTestsFromModule(T)
    r = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
    return len(r.failures) + len(r.errors), r.testsRun

orig_compare = domains._compare
orig_enclose = domains.enclose_normalization
orig_ref_enclose = reference.enclose_normalization
orig_exec = domains.execute_normalization

def m_tolerance():
    def f(box, q64):
        below = q64 < box._lo - 1e-6; above = q64 > box._hi + 1e-6
        return (not (below|above).any()), "mut"
    domains._compare = f
def m_skip_first_coord():
    def f(box, q64):
        below = q64 < box._lo; above = q64 > box._hi
        below[0,0]=False; above[0,0]=False
        return (not (below|above).any()), "mut"
    domains._compare = f
def m_collapse_hi():
    def f(lo, hi, nc):
        a,_ = orig_enclose(lo, hi, nc); return a, a
    domains.enclose_normalization = f
def m_swap():
    def f(lo, hi, nc):
        a,b = orig_enclose(lo, hi, nc); return b, a
    domains.enclose_normalization = f
def m_exec_float64():
    def f(x, nc):
        with np.errstate(all='ignore'):
            return ((x.astype(np.float64)-nc.mean[:,None])/nc.std[:,None]).astype(np.float32)
    domains.execute_normalization = f

muts = {"tolerance 1e-6": m_tolerance, "skip coordinate (0,0)": m_skip_first_coord, "collapse enclosure hi=lo": m_collapse_hi,
        "swap enclosure lo/hi": m_swap, "float64 normalization": m_exec_float64}
print("baseline failures/tests:", run())
for name, m in muts.items():
    domains._compare = orig_compare; domains.enclose_normalization = orig_enclose; domains.execute_normalization = orig_exec
    m()
    f, n = run()
    print(f"mutant [{name}] -> failures={f}/{n}  {'KILLED' if f else 'SURVIVED'}")
