"""compare.py: the planner's grasp / pour searches with planner/cuda/libikchain.so against the numpy loop, on six
table layouts. Run where the library is built (the AGX Orin): python3 planner/cuda/compare.py. Every line should say
same=True (same candidate, joints within 0.02 deg modulo 360). 2026-11-26 on the AGX: 18/18 same, total 20.5 s on
the GPU path vs 40.5 s numpy (a hard grasp search 16.2 s -> 0.6 s)."""
import os, sys, time, importlib, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import ik_cuda, plan77
def run(use_cuda, fn, *a, **k):
    ik_cuda._tried=False; ik_cuda._lib=None
    os.environ["MYCOBOT_IK_CUDA"]="1" if use_cuda else "0"
    t=time.perf_counter()
    try: r=fn(*a,**k)
    except plan77.PlanError as e: r=str(e)[:80]
    return r, time.perf_counter()-t
def same(a,b,keys):
    if isinstance(a,str) or isinstance(b,str): return a==b if isinstance(a,str) and isinstance(b,str) else False
    ok=all(a[k]==b[k] for k in keys)
    for k in ("qa","q","q40"):
        if k in a: ok&=float(np.abs((np.asarray(a[k])-np.asarray(b[k])+180)%360-180).max())<2e-2
    return ok
scenes=[((150,-160),(202,47.5),[]),((145,-155),(220,0),[[228,-89,262.5]]),((133,-202),(222,48),[[228,-89,262.5]]),((194,-155),(202,47),[[178.8,-68.2,262.5]]),((170,-90),(234,23),[[130,-210,262.5]]),((160,-120),(210,30),[])]
plan77.ik_cuda.chain(np.zeros((2,1,4,4))+np.eye(4), np.zeros((2,1,6)))  # warm the GPU
tg=tc=0; KG=("key",); KP=("spin","roll")
for V,C,O in scenes:
    sc=plan77.Scene(np.array(C,float),286.0,40.0,O)
    for pitches in ((0.0,5.0,8.0,12.0),):
        (gc_,t1),(gn,t2)=run(True,plan77.plan_grasp,np.array(V,float),sc,None,None,pitches=pitches),run(False,plan77.plan_grasp,np.array(V,float),sc,None,None,pitches=pitches)
        tg+=t1; tc+=t2
        print(f"grasp V{V}: cuda {t1:.2f}s numpy {t2:.2f}s same={same(gc_,gn,KG)}", gc_["key"] if isinstance(gc_,dict) else gc_)
    for pitch in (0.0,8.0):
        (pc,t1),(pn,t2)=run(True,plan77.plan_pour,np.array(C,float),sc,None,None,pitch=pitch),run(False,plan77.plan_pour,np.array(C,float),sc,None,None,pitch=pitch)
        tg+=t1; tc+=t2
        print(f"  pour C{C} pitch {pitch}: cuda {t1:.2f}s numpy {t2:.2f}s same={same(pc,pn,KP)}", (pc["spin"],pc["roll"]) if isinstance(pc,dict) else pc)
print(f"total cuda {tg:.1f} s, numpy {tc:.1f} s")
