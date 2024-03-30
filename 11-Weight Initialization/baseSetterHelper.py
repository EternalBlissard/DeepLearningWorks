
import os
import numpy as np
import random
import torch
from distutils.version import LooseVersion as Version

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