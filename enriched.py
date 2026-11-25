from pathlib import Path
import zipfile,pandas as pd,numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,brier_score_loss
import torch
from torch import nn
p=Path(__file__).parent;torch.set_num_threads(1)
D=[]
for year,w,edu,housing,rooms in [(2018,4,'s2aq6','sect11_plantingw4.dta','s11q9'),(2023,5,'s2q6','sect9_harvestw5.dta','s9q12')]:
 with zipfile.ZipFile(next(p.glob(f'NGA_{year}*Stata.zip'))) as z:
  def read(n,c):return pd.read_stata(z.open(n),columns=c,convert_categoricals=False)
  d=read('totcons_final.dta',['hhid','zone','sector','ea','hhsize','popw','totcons_adj']); e=read(f'sect2_harvestw{w}.dta',['hhid',edu]); h=read(housing,['hhid',rooms]);e['school']=e[edu].map({1:1,2:0}); e=e.groupby('hhid').school.mean().reset_index();h=h.rename(columns={rooms:'rooms'});h.loc[~h.rooms.between(1,100),'rooms']=np.nan
  d=d.merge(e,on='hhid',how='left',validate='one_to_one').merge(h,on='hhid',how='left',validate='one_to_one');d['crowding']=d.hhsize/d.rooms
  x=d.totcons_adj.to_numpy();wt=d.popw.to_numpy();i=np.argsort(x);med=x[i][np.searchsorted(np.cumsum(wt[i]),wt.sum()/2)];d['target']=(x<.6*med).astype(int);D.append(d);print(year,len(d),d[['school','rooms','crowding']].isna().sum().to_dict())
a,b=D; rows=[]; predictions={}; y=b.target.to_numpy();v=b.popw.to_numpy();actual=np.average(y,weights=v)
for regime,train in [('full',a),('nonoverlap',a[~a.hhid.isin(b.hhid)])]:
 for feature,num in [('basic',['hhsize']),('enriched',['hhsize','school','rooms','crowding'])]:
  pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['zone','sector']),('num',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler()),num)]); X=pre.fit_transform(train);T=pre.transform(b);Y=train.target.to_numpy();W=train.popw.to_numpy();W=W/W.mean()
  models={'LR':LogisticRegression(max_iter=1000),'RF':RandomForestClassifier(n_estimators=200,min_samples_leaf=20,random_state=42,n_jobs=2)}
  preds={}
  for name,m in models.items():m.fit(X,Y,sample_weight=W);preds[name]=m.predict_proba(T)[:,1]
  xt=torch.tensor(X,dtype=torch.float32);tt=torch.tensor(T,dtype=torch.float32);yt=torch.tensor(Y,dtype=torch.float32);wt=torch.tensor(W,dtype=torch.float32);ens=[]
  for seed in range(42,47):
   torch.manual_seed(seed);m=nn.Sequential(nn.Linear(X.shape[1],16),nn.ReLU(),nn.Linear(16,8),nn.ReLU(),nn.Linear(8,1));opt=torch.optim.AdamW(m.parameters(),lr=.005,weight_decay=.01)
   for _ in range(200):
    opt.zero_grad();loss=(nn.functional.binary_cross_entropy_with_logits(m(xt).flatten(),yt,reduction='none')*wt).mean();loss.backward();opt.step()
   with torch.no_grad():ens.append(torch.sigmoid(m(tt)).flatten().numpy())
  preds['DNN']=np.mean(ens,axis=0)
  for name,q in preds.items():
   est=np.average(q,weights=v);rows.append(dict(regime=regime,features=feature,model=name,train_n=len(train),auc=roc_auc_score(y,q,sample_weight=v),brier=brier_score_loss(y,q,sample_weight=v),estimate=est,actual=actual,error_pp=100*abs(est-actual)));predictions[regime,feature,name]=q
r=pd.DataFrame(rows);r.to_csv(p/'enriched_results.csv',index=False);print(r.to_string(index=False))
# paired cluster bootstrap of fixed predictions, stratified by zone/sector
rng=np.random.default_rng(42); groups=[]
for _,g in b.groupby(['zone','sector']):groups.append([x.index.to_numpy() for _,x in g.groupby('ea')])
q=predictions['full','enriched','DNN'];s=predictions['full','enriched','RF'];diff=[]
for _ in range(500):
 ix=np.concatenate([np.concatenate([g[j] for j in rng.integers(len(g),size=len(g))]) for g in groups]);diff.append([roc_auc_score(y[ix],q[ix],sample_weight=v[ix])-roc_auc_score(y[ix],s[ix],sample_weight=v[ix]),np.average((y[ix]-q[ix])**2-(y[ix]-s[ix])**2,weights=v[ix])])
ci=pd.DataFrame(np.quantile(diff,[.025,.975],axis=0),index=['lower','upper'],columns=['auc_difference','brier_difference']);ci.to_csv(p/'enriched_bootstrap.csv');print(ci)
