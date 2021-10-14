import os
import sys
import numpy as np
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import GUINetwork as Network

#from sklearn.metrics import precision_recall_fscore_support as score, plot_confusion_matrix
import sklearn.metrics 
import matplotlib.pyplot as plt
import itertools


parser = argparse.ArgumentParser("GUI-data")
parser.add_argument('--data', type=str, default='./train_data/GUI-data/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--model_path', type=str, default='./model_path/model_best.pt', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='NASGW', help='which architecture to use')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

CLASSES = 15


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
  model = model.cuda()
  model.load_state_dict(torch.load(args.model_path)['state_dict'])

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  validdir = os.path.join(args.data, 'test')
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  valid_data = dset.ImageFolder(
    validdir,
    transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
    ]))
  class_names = ['Bt', 'CB', 'CTV', 'ET', 'IBt', 'IV', 'NP', 'PB', 'RBt', 'RB', 'SB', 'Sp', 'Sw', 'TV', 'TBt']

  valid_queue = torch.utils.data.DataLoader(
    valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

  model.drop_path_prob = args.drop_path_prob
  #valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)

  model.eval()
  predlist=torch.zeros(0,dtype=torch.long, device='cpu')
  lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
  #......................Added..............................
  for step, (input, target) in enumerate(valid_queue):
    input = input.cuda()
    target = target.cuda(non_blocking=True)
    with torch.no_grad():
      logits, _ = model(input)
      loss = criterion(logits, target)
      _, preds = torch.max(logits, -1)

      # Append batch prediction results
      lbllist=torch.cat([lbllist,target.view(-1).cpu()])
      predlist=torch.cat([predlist,preds.view(-1).cpu()])
      

  print()
  print()
  print ('Accuracy')
  print (sklearn.metrics.accuracy_score(lbllist.numpy(),predlist.numpy()))
  print()
  print ('Precision')
  print (sklearn.metrics.precision_score(lbllist.numpy(),predlist.numpy(), average='weighted'))
  print()
  print ('Recall')
  print (sklearn.metrics.recall_score(lbllist.numpy(),predlist.numpy(), average='weighted'))
  print()
  print ('F1-Score')
  print (sklearn.metrics.f1_score(lbllist.numpy(),predlist.numpy(), average='weighted'))

 

if __name__ == '__main__':
  main() 
