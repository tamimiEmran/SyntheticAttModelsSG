from imblearn.over_sampling import KMeansSMOTE
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import main_evaluate as me


def _resample(allExamples, allLabels):
    sm = KMeansSMOTE(
            kmeans_estimator=MiniBatchKMeans(n_init=1, random_state=0), random_state=0
        )
    return sm.fit_resample(allExamples, allLabels)


def _examplesForResampling(benignExamples, modifiedExamples):
    label_normal = np.full((1, benignExamples.shape[1]), 0)
    label_theft = np.full((1, modifiedExamples.shape[1]), 1)
    
    labels = np.vstack((label_normal, label_theft))
    
    tot_examples = np.vstack((benignExamples, modifiedExamples))
    
    return tot_examples, labels.flatten()


def oversample(bExamples, modifiedExamples):
    return _resample(*_examplesForResampling(bExamples, modifiedExamples))



    

o = me.loadOriginalExamples()[0]
np.random.shuffle(o)
o = o[:len(o)//2]

t = me.load_theft_examples()

y_o = np.array([0] * len(o)).astype(np.int32).reshape(-1,1)
y_t = np.array([1] * len(t)).astype(np.int32).reshape(-1,1)

    
allExamples = np.vstack((o,t))
allLabels = np.vstack((y_o, y_t))




data = _resample(allExamples, allLabels)
    
    
    
    
    
    