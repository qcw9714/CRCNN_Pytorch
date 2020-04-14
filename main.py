import torch
import pandas as pd
import numpy as np
import csv
import spacy
import os
import re
from torchtext import data, datasets, vocab
import argparse
import train as trains
import model
import datetime
from torchtext.data import Iterator, BucketIterator
print('parse arguments.')
parser = argparse.ArgumentParser(description='CRCNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.025, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=128, help='number of epochs for train [default: 16]')
parser.add_argument('-batch-size', type=int, default=128, help='batch size for training [default: 256]')
parser.add_argument('-log-interval',  type=int, default=200,   help='how many steps to wait before logging training status [default: 500]')
parser.add_argument('-dev-interval', type=int, default=400, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=600, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=2000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.25, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=100, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=600, help='number of each kind of kernel')
#parser.add_argument('-kernel-sizes', type=str, default='2,3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-kernel-sizes', type=str, default='3', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
  print("\t{}={}".format(attr.upper(),value))

#args.sent_len = 90
args.class_num = 19
args.pos_dim = 80
args.mPos = 2.5
args.mNeg = 0.5
args.gamma = 0.05
# args.device = torch.device(args.device)
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]

nlp = spacy.load('en_core_web_sm')

def tokenizer(text): # create a tokenizer function
  return [tok.text for tok in nlp.tokenizer(text)]

def emb_tokenizer(l):
  r = [y for x in eval(l) for y in x]
  return r
def y_tokenize(y):
  return int(y)
TEXT = data.Field(sequential=True, tokenize=tokenizer,batch_first=True)
#LABEL = data.Field(sequential=False, use_vocab=True,batch_first=True)
LABEL = data.ReversibleField(sequential=False, unk_token='OTHER',use_vocab=True,batch_first=True)
POS_EMB = data.Field(sequential=True, tokenize=emb_tokenizer,batch_first=True)

print('loading data...')
train,valid,test = data.TabularDataset.splits(path='../data/SemEval2010_task8_all_data',
  train='SemEval2010_task8_training/TRAIN_FILE_SUB.CSV',
  validation='SemEval2010_task8_training/VALID_FILE.CSV',
  test='SemEval2010_task8_testing_keys/TEST_FILE_FULL.CSV',
  format='csv',
  skip_header=True,csv_reader_params={'delimiter':'\t'},
  fields=[('relation',LABEL),('sentence',TEXT),('pos_embed',POS_EMB)])
print('load data end')
#print(valid[0].__dict__)
#print(type(valid[0].relation))
#print(valid[1].__dict__)
#print(type(valid[1].pos_embed[0]))

vectorsuse = vocab.Vectors(name='../glove.6B/glove.6B.100d.txt')
TEXT.build_vocab(train,vectors=vectorsuse)
LABEL.build_vocab(train)
POS_EMB.build_vocab(train)

allposemb = []
alllabelvo = []


args.vocab = TEXT.vocab
print(len(args.vocab))
print(args.vocab.itos[0])
print(args.vocab.itos[1])
print(args.vocab.itos[2])
args.posemb = POS_EMB.vocab
for i in range(0,len(args.posemb)):
  if args.posemb.itos[i] not in allposemb:
    allposemb.append(args.posemb.itos[i])

for i in range(0,len(LABEL.vocab)):
  if LABEL.vocab.itos[i] not in alllabelvo:
    alllabelvo.append(LABEL.vocab.itos[i])

print(allposemb)
print(alllabelvo)
#print(len(args.posemb))
#print(len(LABEL.vocab))
#print(LABEL.vocab.itos[0])
#print(LABEL.vocab.itos[1])
#print(LABEL.vocab.itos[2])
#print(args.posemb.itos[0])
#print(args.posemb.itos[159])

args.cuda = torch.cuda.is_available()
print(args.cuda)
# args.cuda = False
args.save_dir = os.path.join(args.save_dir,datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
train_iter, val_iter = BucketIterator.splits((train,valid),batch_sizes=(args.batch_size,len(valid)),
  device=args.device,
  sort_key=lambda x: len(x.sentence),
  sort_within_batch=False,
  repeat=False)


test_iter = Iterator(test,batch_size=len(test),device=args.device,sort=False,sort_within_batch=False,repeat=False)
#for batch in val_iter:
  #print(batch.relation)
  #feature, target, pos = batch.sentence, batch.relation, batch.pos_embed #(W,N) (N)
  #orig_text = LABEL.reverse(target)
  #print(target)
  #print(orig_text)
  #target = target - 1
  #print(target)
print('build model...')
cnn = model.CRCNN(args)
if args.snapshot is not None:
  print('\nLoding model from {}...'.format(args.snapshot))
  cnn.load_state_dict(torch.load(args.snapshot))

if args.cuda:

  torch.cuda.set_device(args.device)
  cnn = cnn.cuda()
  print("aaaaaaaa")

if args.test:
  try:
    trains.eval(test_iter,cnn,args)
  except Exception as e:
    print("\n test wrong.")
else:
  print("bbbbbbbbb")
  trains.train(train_iter,val_iter,cnn,args)
