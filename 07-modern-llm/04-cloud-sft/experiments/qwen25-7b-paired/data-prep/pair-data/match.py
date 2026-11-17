"""Select disjoint real pairs with exactly matching input-token totals in one quarter."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
rows=json.loads(Path(sys.argv[1]).read_text())
vertices={x:i for i,x in enumerate(sorted({v for a,b,d,q in rows for v in (a,b)}))}
n=len(rows);m=len(vertices)
i=[];j=[];v=[]
for k,(a,b,delta,quality) in enumerate(rows):
    i.extend([vertices[a],vertices[b],m]); j.extend([k,k,k]); v.extend([1,1,sum(delta)])
matrix=coo_matrix((v,(i,j)),shape=(m+1,n)).tocsc()
objective=-np.array([1+q*.002 for a,b,delta,q in rows])
z=milp(objective,integrality=np.ones(n,dtype=np.uint8),bounds=Bounds(0.,1.),
       constraints=LinearConstraint(matrix,np.zeros(m+1),np.r_[np.ones(m),0.]),
       options={'time_limit':120,'mip_rel_gap':.022,'presolve':True})
if z.x is None:raise ValueError(z.message)
chosen=np.flatnonzero(z.x>.5).tolist()
assert sum(sum(rows[k][2]) for k in chosen)==0
assert len({t for k in chosen for t in rows[k][:2]})==len(chosen)*2
Path(sys.argv[2]).write_text(json.dumps(chosen))
print(Path(sys.argv[1]).name,len(rows),'edges',len(chosen),'pairs',z.message,flush=True)
