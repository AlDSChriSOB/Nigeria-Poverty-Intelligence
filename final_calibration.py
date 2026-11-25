from pathlib import Path
exec(Path(__file__).with_name('enriched.py').read_text().split('a,b=D;')[0])
from sklearn.model_selection import GroupKFold
from scipy.special import expit,logit
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score,brier_score_loss
from sklearn.base import clone
a,b=D; y=b.target.to_numpy(); v=b.popw.to_numpy(); Y=a.target.to_numpy(); W=a.popw.to_numpy(); cols=['hhsize','school','rooms','crowding']
pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['zone','sector']),('num',StandardScaler(),cols)])
# Cross-fitted source predictions; no target outcomes used for calibration.
oof=np.zeros(len(a)); preds=[]
for seed,(tr,va) in enumerate(GroupKFold(5).split(a,Y,a.ea)):
 X=pre.fit_transform(a.iloc[tr]); T=pre.transform(b); H=pre.transform(a.iloc[va]); torch.manual_seed(seed)
 m=nn.Sequential(nn.Linear(X.shape[1],16),nn.ReLU(),nn.Dropout(.1),nn.Linear(16,8),nn.ReLU(),nn.Linear(8,1)); opt=torch.optim.AdamW(m.parameters(),lr=.005)
 x=torch.tensor(X,dtype=torch.float32); t=torch.tensor(Y[tr],dtype=torch.float32); w=torch.tensor(W[tr]/W[tr].mean(),dtype=torch.float32)
 for _ in range(200):
  opt.zero_grad(); loss=(nn.functional.binary_cross_entropy_with_logits(m(x).flatten(),t,reduction='none')*w).mean();loss.backward();opt.step()
 m.eval()
 with torch.no_grad():
  oof[va]=torch.sigmoid(m(torch.tensor(H,dtype=torch.float32))).flatten().numpy();preds.append(torch.sigmoid(m(torch.tensor(T,dtype=torch.float32))).flatten().numpy())
q=np.mean(preds,axis=0); clip=lambda x:np.clip(x,1e-6,1-1e-6)
shift=brentq(lambda c:np.average(expit(logit(clip(oof))+c),weights=W)-np.average(Y,weights=W),-20,20)
r=expit(logit(clip(q))+shift); rng=np.random.default_rng(42); groups=[[x.index.to_numpy() for _,x in g.groupby('ea')] for _,g in b.groupby(['zone','sector'])]; rows=[]
for name,pred in [('uncalibrated',q),('source_calibrated',r)]:
 vals=[]
 for _ in range(500):
  ix=np.concatenate([np.concatenate([g[j] for j in rng.integers(len(g),size=len(g))]) for g in groups]); vals.append(np.average(pred[ix]-y[ix],weights=v[ix]))
 rows.append(dict(model=name,auc=roc_auc_score(y,pred,sample_weight=v),brier=brier_score_loss(y,pred,sample_weight=v),estimate=np.average(pred,weights=v),actual=np.average(y,weights=v),signed_error_ci_low=np.quantile(vals,.025),signed_error_ci_high=np.quantile(vals,.975),source_shift=shift))
res=pd.DataFrame(rows);res.to_csv(p/'final_calibration_results.csv',index=False);print(res.to_string(index=False))
