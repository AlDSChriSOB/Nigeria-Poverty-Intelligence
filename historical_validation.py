from pathlib import Path
import pandas as pd,numpy as np,zipfile
p=Path(__file__).parent
rows=[]; waves={}
for year,pattern in [(2019,'NGA_2018*'),(2024,'NGA_2023*')]:
 with zipfile.ZipFile(next(p.glob(pattern+'Stata.zip'))) as z:
  assert z.testzip() is None
  d=pd.read_stata(z.open('totcons_final.dta'),convert_categoricals=False)
 with zipfile.ZipFile(next(p.glob(pattern+'CSV.zip'))) as z:
  assert z.testzip() is None
  c=pd.read_csv(z.open('totcons_final.csv'))
  a=d.merge(c,on='hhid',validate='one_to_one',suffixes=('_s','_c'))
  assert len(a)==len(d)==len(c)
  for col in ['totcons_adj','popw','hhsize']:
   assert np.allclose(a[col+'_s'],a[col+'_c'],rtol=1e-5)
 assert not d.hhid.duplicated().any()
 assert (d[['totcons_adj','popw','hhsize']]>0).all().all()
 waves[year]=d
 def gini(x,w):
  i=np.argsort(x); x=np.asarray(x,dtype=float)[i]; w=np.asarray(w,dtype=float)[i]; cw=np.r_[0,np.cumsum(w)/sum(w)]; cy=np.r_[0,np.cumsum(w*x)/sum(w*x)]; return 1-2*np.trapezoid(cy,cw)
 for label,m in [('National',d),('Urban',d[d.sector==1]),('Rural',d[d.sector==2])]:
  rows.append(dict(year=year,domain=label,n=len(m),gini=gini(m.totcons_adj,m.popw),mean_local=np.average(m.totcons_adj,weights=m.popw)))
r=pd.DataFrame(rows); r.to_csv(p/'historical_validation.csv',index=False); print(r.to_string(index=False))
# Scale-free distributional persistence benchmark, no later outcomes used in prediction.
a=waves[2019]; b=waves[2024]
for k,d in waves.items():
 y=d.totcons_adj.to_numpy(); w=d.popw.to_numpy(); i=np.argsort(y); med=y[i][np.searchsorted(np.cumsum(w[i])/w.sum(),.5)]; print(k,'relative poverty',np.average(y<.6*med,weights=w),'median',med)
