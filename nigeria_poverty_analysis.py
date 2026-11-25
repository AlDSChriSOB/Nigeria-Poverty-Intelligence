import pandas as pd, numpy as np, zipfile
from pathlib import Path
base=Path(__file__).parent
with zipfile.ZipFile(base/'NGA_2023_GHSP-W5_v03_M_CSV.zip') as z:
 d=pd.read_csv(z.open('totcons_final.csv')); a=pd.read_csv(z.open('secta_harvestw5.csv'))
with zipfile.ZipFile(base/'NGA_2023_GHSP-W5_v03_M_Stata.zip') as z:
 s=pd.read_stata(z.open('totcons_final.dta'),convert_categoricals=False)
assert np.allclose(d.sort_values('hhid').totcons_adj21,s.sort_values('hhid').totcons_adj21)
m=d.merge(a[['hhid','wt_cross_wave5','cluster','strata']],on='hhid',validate='one_to_one'); assert len(m)==len(d)==4715
assert (m[['hhsize','wt_cross_wave5','totcons_adj21']]>0).all().all()
m['w']=m.hhsize*m.wt_cross_wave5
m['poor']=m.totcons_adj21<3
m['zone']=m.zone.map(dict(enumerate(['North Central','North East','North West','South East','South South','South West'],1)))
m['sector']=m.sector.map({1:'Urban',2:'Rural'})
rows=[]
for dimension, groups in [('National',[('Nigeria',m)]),('Zone',m.groupby('zone')),('Sector',m.groupby('sector'))]:
 for name,g in groups:
  rows.append(dict(dimension=dimension,group=name,households=len(g),poverty_pct=100*np.average(g.poor,weights=g.w),mean_daily_ppp=np.average(g.totcons_adj21,weights=g.w)))
out=pd.DataFrame(rows); out.to_csv(base/'nigeria_poverty_preliminary.csv',index=False); print(out.to_string(index=False))
x=m.sort_values('totcons_adj21'); w=x.w.to_numpy(); y=x.totcons_adj21.to_numpy(); X=np.r_[0,np.cumsum(w)/w.sum()]; Y=np.r_[0,np.cumsum(w*y)/sum(w*y)]; print('Gini',1-2*np.trapezoid(Y,X)); print('threshold sensitivity',[(t,100*np.average(m.totcons_adj21<t,weights=m.w)) for t in [2.5,3,3.5,4.2]])
