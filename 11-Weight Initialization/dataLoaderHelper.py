import torch
from torch.utils.data import sampler
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms

def getDataLoadersMNIST(batchSize, numWorkers=0,validFraction =None, trainTransforms =None, testTransforms =None ):
  if(trainTransforms is None):
    trainTransforms = transforms.ToTensor()
  if(testTransforms is None):
    testTransforms = transforms.ToTensor()

  trainDataset = datasets.MNIST(root='data',
                              train=True,
                              transform=trainTransforms,
                              download = True)

  testDataset = datasets.MNIST(root='data',
                              train=True,
                              transform=testTransforms)

  testDataset = datasets.MNIST(root='data',
                              train=False,
                              transform=testTransforms)

  if(validFraction is not None):
    num = int(validFraction*60000)
    trainIndices = torch.arange(0,60000-num)
    valIndices = torch.arange(60000-num, 60000)

    trainSampler=SubsetRandomSampler(trainIndices)
    valSampler = SubsetRandomSampler(valIndices)

    trainLoader = DataLoader(dataset=trainDataset,
                         batch_size=batchSize,
                         num_workers=numWorkers,
                         drop_last = True,
                         sampler = trainSampler)
    valLoader = DataLoader(dataset=trainDataset,
                         batch_size=batchSize,
                         num_workers=numWorkers,
                         sampler = valSampler)
  else:
    trainLoader = DataLoader(dataset=trainDataset,
                         batch_size=batchSize,
                         num_workers=numWorkers,
                         drop_last = True,
                         shuffle=True)

  testLoader = DataLoader(dataset=testDataset,
                         batch_size=batchSize,
                         shuffle=False,
                         num_workers=numWorkers)

  if(validFraction is None):
    return trainLoader,testLoader
  else:
    return trainLoader,valLoader,testLoader