from pathlib import Path
# Reuse the exact harmonisation stage, not the previous model fitting.
exec(Path(__file__).with_name('enriched.py').read_text().split('a,b=D;')[0])
a,b=D; rows=[]
class Reverse(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x): return x.view_as(x)
 @staticmethod
 def backward(ctx,g): return -g
class FT(nn.Module):
 def __init__(self,n):
  super().__init__(); self.w=nn.Parameter(torch.randn(n,8)*.1); self.bias=nn.Parameter(torch.zeros(n,8)); self.cls=nn.Parameter(torch.zeros(1,1,8)); self.enc=nn.TransformerEncoderLayer(8,2,16,dropout=.1,batch_first=True); self.out=nn.Linear(8,1)
 def forward(self,x):
  t=x[:,:,None]*self.w+self.bias; return self.out(self.enc(torch.cat([self.cls.expand(len(x),-1,-1),t],1))[:,0]).flatten()
for regime,train in [('full',a),('nonoverlap',a[~a.hhid.isin(b.hhid)])]:
 pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['zone','sector']),('num',StandardScaler(),['hhsize','school','rooms','crowding'])]); X=torch.tensor(pre.fit_transform(train),dtype=torch.float32); T=torch.tensor(pre.transform(b),dtype=torch.float32); Y=torch.tensor(train.target.values,dtype=torch.float32); W=torch.tensor(train.popw.values/train.popw.mean(),dtype=torch.float32); V=torch.tensor(b.popw.values/b.popw.mean(),dtype=torch.float32)
 for name in ['DANN','FT_style']:
  preds=[]
  for seed in [42,43,44]:
   torch.manual_seed(seed); enc=nn.Sequential(nn.Linear(X.shape[1],16),nn.ReLU(),nn.Linear(16,8),nn.ReLU()); head=nn.Linear(8,1); dom=nn.Linear(8,1); ft=FT(X.shape[1]); params=list(ft.parameters()) if name=='FT_style' else list(enc.parameters())+list(head.parameters())+list(dom.parameters()); opt=torch.optim.AdamW(params,lr=.005,weight_decay=.01); loss=nn.BCEWithLogitsLoss(reduction='none')
   for epoch in range(80):
    ix=torch.randperm(len(X))[:512]; jx=torch.randperm(len(T))[:512]; opt.zero_grad()
    if name=='FT_style': l=(loss(ft(X[ix]),Y[ix])*W[ix]).mean()
    else:
     h=enc(X[ix]); t=enc(T[jx]); l=(loss(head(h).flatten(),Y[ix])*W[ix]).mean()+.1*((loss(dom(Reverse.apply(h)).flatten(),torch.zeros(len(ix)))*W[ix]).mean()+(loss(dom(Reverse.apply(t)).flatten(),torch.ones(len(jx)))*V[jx]).mean())/2
    l.backward();opt.step()
   ft.eval()
   with torch.no_grad(): preds.append(torch.sigmoid(ft(T) if name=='FT_style' else head(enc(T)).flatten()).numpy())
  q=np.mean(preds,axis=0); y=b.target; v=b.popw; est=np.average(q,weights=v); actual=np.average(y,weights=v); rows.append(dict(regime=regime,model=name,auc=roc_auc_score(y,q,sample_weight=v),brier=brier_score_loss(y,q,sample_weight=v),estimate=est,actual=actual,error_pp=100*abs(est-actual),seed_sd_pp=100*np.std([np.average(s,weights=v) for s in preds])))
r=pd.DataFrame(rows);r.to_csv(p/'advanced_poverty_results.csv',index=False);print(r.to_string(index=False))
