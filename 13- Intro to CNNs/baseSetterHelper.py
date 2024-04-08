
import os
import numpy as np
import random
import torch
from distutils.version import LooseVersion as Version
from itertools import product

def setAllSeeds(seed):
  os.environ['MY_GLOBAL_SEED'] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def setDeterministic():
  if (torch.cuda.is_available()):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  if (torch.__version__ <= Version("1.7")):
    torch.set_deterministic(True)
  else:
    torch.use_deterministic_algorithms(True)

def computeAccu(model,dataLoader,device):
  model.eval()
  with torch.no_grad():
    correctPred = 0
    totalPred = 0

    for features,targets in dataLoader:
      features = features.to(device)
      targets  = targets.to(device)

      logits   = model(features)
      _ , predLabel = torch.max(logits,1)
      totalPred += targets.size(0)
      correctPred += (predLabel == targets).sum()

  return correctPred.float()/totalPred * 100

def computeConfusionMatrix(model,dataLoader,device):
  allTargets = []
  allPredictions = []

  with torch.no_grad():
    for i, (features,targets) in enumerate(dataLoader):
      features = features.to(device)
      targets = targets.to(device)
      logits = model(features)
      _, predLabels = torch.max(logits,1)
      allTargets.extend(targets.to('cpu'))
      allPredictions.extend(predLabels.to('cpu'))

  allTargets = np.array(allTargets)
  allPredictions = np.array(allPredictions)
  
  classLabels = np.unique(np.concatenate((allTargets,allPredictions)))
  if (classLabels.shape[0]==1):
    if(classLabels[0]!=0):
      classLabels = np.array([0,classLabels[0]])
    else:
      classLabels = np.array([classLabels[0],1])
  nLabels = classLabels.shape[0]
  lst = []
  z = list(zip(allTargets, allPredictions))
  for combi in product(classLabels, repeat=2):
      lst.append(z.count(combi))
  mat = np.asarray(lst)[:, None].reshape(nLabels, nLabels)
  return mat