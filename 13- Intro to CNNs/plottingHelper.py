import os
import matplotlib.pyplot as plt
import numpy as np
import torch

def plotTrainingLoss(miniBatchLoss,numEpoch,iterPerEpoch,resultsDir=None,avgIter = 100):
  plt.figure()
  ax1 = plt.subplot(1,1,1)
  ax1.plot(range(len(miniBatchLoss)), miniBatchLoss, label='Mini Batch Loss')
  if len(miniBatchLoss) > 1000:
    ax1.set_ylim([0,np.max(miniBatchLoss[1000:])*1.5])
  ax1.set_xlabel('Iterations')
  ax1.set_ylabel('Loss')
  ax1.plot(np.convolve(miniBatchLoss,np.ones(avgIter,)/avgIter,mode='valid'),label='Running Avg')
  ax1.legend()

  ax2 = ax1.twiny()
  newLabel = list(range(numEpoch+1))
  newPos = [e*iterPerEpoch for e in newLabel]
  # ax2.set_xticks(newpos[::10])
  # ax2.set_xticklabels(newlabel[::10])

  ax2.set_xticks(newPos[::10])
  ax2.set_xticklabels(newLabel[::10])
  ax2.spines['bottom'].set_position(('outward',45))
  ax2.set_xlabel("Epochs")
  ax2.set_xlim(ax1.get_xlim())

  plt.tight_layout()

  if(resultsDir is not None):
    imagePath = os.path.join(resultsDir, 'plotTrainingLoss.pdf')
    plt.savefig(imagePath)

def plotAccuracy(trainAccList, valAccList, resultsDir = None):
  numEpoch = len(trainAccList)
  plt.plot(np.arange(1,numEpoch+1),trainAccList,label='Training')
  plt.plot(np.arange(1,numEpoch+1),valAccList,label='Validation')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()

  plt.tight_layout()


  if(resultsDir is not None):
    imagePath = os.path.join(resultsDir, 'plotAccTrainingValidation.pdf')
    plt.savefig(imagePath)

def show_examples(model, dataLoader):
  for batchIdx, (features, targets) in enumerate(dataLoader):
    with torch.no_grad():
      features = features.to(torch.device('cpu'))
      targets  = targets.to(torch.device('cpu'))
      logits = model(features)
      predictions = torch.argmax(logits,dim=1)
    break

  fig, axes = plt.subplots(nrows=3,ncols=5,sharex=True,sharey=True)
  nhwcImage = np.transpose(features,axes = (0,2,3,1))
  nhwImage  = np.squeeze(nhwcImage.numpy(), axis=3)

  for idx,ax in enumerate(axes.ravel()):
    ax.imshow(nhwImage[idx],cmap='binary')
    ax.title.set_text(f'P: {predictions[idx]} | T: {targets[idx]}')
    ax.axison = False

  plt.tight_layout()
  plt.show()

def plotConfusionMat(confMat,hideSpines=False,hideTicks=False,figSize=None,cmap=None,colorbar=False,showAbsolute=True,showNormed=False,classNames=None):
  if not (showAbsolute or showNormed):
        raise AssertionError('Both show_absolute and show_normed are False')
  if classNames is not None and len(classNames) != len(confMat):
        raise AssertionError('len(classNames) should be equal to number of'
                             'classes in the dataset')
  
  totalSamples = confMat.sum(axis=1)[:, np.newaxis]
  normedConfMat = confMat.astype('float') / totalSamples

  fig, ax = plt.subplots(figsize=figSize)
  ax.grid(False)
  if cmap is None:
      cmap = plt.cm.Blues

  if figSize is None:
      figSize = (len(confMat)*1.25, len(confMat)*1.25)

  if showNormed:
      matshow = ax.matshow(normedConfMat, cmap=cmap)
  else:
      matshow = ax.matshow(confMat, cmap=cmap)

  if colorbar:
      fig.colorbar(matshow)
  
  for i in range(confMat.shape[0]):
    for j in range(confMat.shape[1]):
        cell_text = ""
        if showAbsolute:
            cell_text += format(confMat[i, j], 'd')
            if showNormed:
                cell_text += "\n" + '('
                cell_text += format(normedConfMat[i, j], '.2f') + ')'
        else:
            cell_text += format(normedConfMat[i, j], '.2f')
        ax.text(x=j,
                y=i,
                s=cell_text,
                va='center',
                ha='center',
                color="white" if normedConfMat[i, j] > 0.5 else "black")

  if classNames is not None:
      tickMarks = np.arange(len(classNames))
      plt.xticks(tickMarks, classNames, rotation=90)
      plt.yticks(tickMarks, classNames)
      
  if hideSpines:
      ax.spines['right'].set_visible(False)
      ax.spines['top'].set_visible(False)
      ax.spines['left'].set_visible(False)
      ax.spines['bottom'].set_visible(False)
  ax.yaxis.set_ticks_position('left')
  ax.xaxis.set_ticks_position('bottom')
  if hideTicks:
      ax.axes.get_yaxis().set_ticks([])
      ax.axes.get_xaxis().set_ticks([])

  plt.xlabel('predicted label')
  plt.ylabel('true label')
  return fig, ax












