from olorenchemengine.internal import *
from olorenchemengine.representations import *

class SimLookup_beta(BaseCompoundVecRepresentation):
    
    @log_arguments
    def __init__(self, rep: BaseCompoundVecRepresentation, *args, metric: str = "cosine", 
                 k: int = 3, return_reps: bool = False, log: bool = True, **kwargs):
        self.rep = rep
        self.metric = metric
        self.k = k
        self.return_reps = return_reps
        
        self.fitted = False
        super().__init__(*args, log=False, **kwargs)
    
    def _convert(self):
        assert False, "Directly accessing _convert is not implemented with SimLookup. Use convert instead."
        
    def convert(self, Xs, ys = None, fit = False, **kwargs):
        Xs = SMILESRepresentation().convert(Xs)
        
        assert fit or self.fitted, "SimLookup has not been fitted yet. Call convert with fit=True first."
        
        if fit:
            self.Xs = Xs
            self.ys = ys
            dists = self.rep.calculate_distance(Xs, Xs, metric = self.metric)
            dists = dists + np.identity(dists.shape[0])
            best_ix = np.argpartition(dists, self.k, axis=1)[:, :self.k]
            feats = []
            for i in range(len(Xs)):
                if self.return_reps:
                    feats.append(np.concatenate([dists[i][best_ix[i]], ys[best_ix[i]], self.rep.convert([self.Xs[i]])[0]]))
                else:
                    feats.append(np.concatenate([dists[i][best_ix[i]], ys[best_ix[i]]]))
            self.fitted = True
            return feats
        else:
            dists = self.rep.calculate_distance(Xs, self.Xs, metric = self.metric)
            best_ix = np.argpartition(dists, self.k, axis=1)[:, :self.k]
            feats = []
            for i in range(len(Xs)):
                if self.return_reps:
                    feats.append(np.concatenate([dists[i][best_ix[i]], self.ys[best_ix[i]], self.rep.convert([self.Xs[i]])[0]]))
                else:
                    feats.append(np.concatenate([dists[i][best_ix[i]], self.ys[best_ix[i]]]))
            return feats
    
    def _save(self):
        d = super()._save()
        d.update({"fitted", self.fitted,})
        if hasattr(self, "Xs"):
            d.update({"Xs": self.Xs, "ys": self.ys})
        return d
    
    def _load(self, d):
        super()._load(d)
        self.fitted = d["fitted"]
        if "Xs" in d:
            self.Xs = d["Xs"]
            self.ys = d["ys"]
            