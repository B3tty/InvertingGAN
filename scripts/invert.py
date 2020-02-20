import sys
sys.path.append('../')

from dataload import CELEBA, SHOES, OMNI
from utils import make_new_folder, plot_norm_losses, save_input_args, \
sample_z, class_loss_fn, plot_losses, corrupt, prep_data, plot_log_losses # one_hot
from models import GEN, DIS, GEN1D, DIS1D


import torch
from torch import optim
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy as bce

from torchvision import transforms
from torchvision.utils import make_grid, save_image

import numpy as np

import os
from os.path import join

import argparse

from PIL import Image

import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from time import time

EPSILON = 1e-6

def get_args():
	parser = argparse.ArgumentParser()
	#parser.add_argument('--root', default='../../../../data/', type=str)
	parser.add_argument('--root', default='../Data/bootsData/', type=str)
	parser.add_argument('--batchSize', default=128, type=int)
	parser.add_argument('--maxEpochs', default=200, type=int)
	parser.add_argument('--nz', default=100, type=int)
	parser.add_argument('--imSize', default=64, type=int)
	parser.add_argument('--lr', default=2e-4, type=float)
	parser.add_argument('--fSize', default=64, type=int)  #multiple of filters to use
	#parser.add_argument('--exDir', required=True, type=str)
	parser.add_argument('--exDir', default='../Data/InvertBoots', type=str)
	parser.add_argument('--gpuNo', default=0, type=int)
	parser.add_argument('--alpha', default=1e-6, type=float)
	parser.add_argument('--data', default='SHOES', type=str)  #CELEBA, SHOES or OMNI
	parser.add_argument('--oneBatch', action='store_true') #to process just one bach

	return parser.parse_args()


def find_z(gen, x, nz, lr, exDir, maxEpochs=100):

	#generator in eval mode
	gen.eval()

	#save the "original" images
	save_image(x.data, join(exDir, 'original.png'), normalize=True)

	if gen.useCUDA:
		gen.cuda()
		Zinit = Variable(torch.randn(x.size(0),nz).cuda(), requires_grad=True)
	else:
		Zinit = Variable(torch.randn(x.size(0),nz), requires_grad=True)

	#optimizer
	optZ = torch.optim.RMSprop([Zinit], lr=lr)

	losses = {'rec': []}
	print("maxepochs: ", maxEpochs)
	for e in range(maxEpochs):

		xHAT = gen.forward(Zinit)
		recLoss = F.mse_loss(xHAT, x)

		optZ.zero_grad()
		recLoss.backward()
		optZ.step()

		losses['rec'].append(recLoss.data.item())
		print('[%d] loss: %0.5f' % (e, recLoss.data.item()))

		#plot training losses
		if e>0:
			plot_losses(losses, exDir, e+1)
			plot_log_losses(losses, exDir, e+1)

	#visualise the final output
	xHAT = gen.forward(Zinit)
	save_image(xHAT.data, join(exDir, 'rec.png'), normalize=True)

	return Zinit


def find_batch_z(gen, x, nz, lr, exDir, maxEpochs=100, alpha=1e-6, batchNo=0):

	#generator in eval mode
	gen.eval()

	#save the "original" images
	save_image(x.data, join(exDir, 'original_batch'+str(batchNo)+'.png'), normalize=True, nrow=10)

	#Assume the prior is Standard Normal
	pdf = torch.distributions.Normal(0, 1)

	if gen.useCUDA:
		Zinit = Variable(torch.randn(x.size(0),opts.nz).cuda(), requires_grad=True)
	else:
		Zinit = Variable(torch.randn(x.size(0),opts.nz), requires_grad=True)

	#optimizer
	optZ = torch.optim.RMSprop([Zinit], lr=lr)

	losses = {'rec': [], 'logProb': []}
	for e in range(maxEpochs):

		#reconstruction loss
		xHAT = gen.forward(Zinit)
		recLoss = F.mse_loss(xHAT, x)

		#loss to make sure z's are Guassian
		logProb = pdf.log_prob(Zinit).mean(dim=1)  #each element of Z is independant, so likelihood is a sum of log of elements
		loss = recLoss - (alpha * logProb.mean())
		

		optZ.zero_grad()
		loss.backward()
		optZ.step()

		losses['rec'].append(recLoss.data.item())
		losses['logProb'].append(logProb.mean().data.item())

		if e%100==0:
			print('[%d] loss: %0.5f, recLoss: %0.5f, regMean: %0.5f' % (e, loss.data.item(), recLoss.data.item(), logProb.mean().data.item()))
			# save_image(xHAT.data, join(exDir, 'rec'+str(e)+'.png'), normalize=True)

		#plot training losses
		if e>0:
			plot_losses(losses, exDir, e+1)
			plot_norm_losses(losses, exDir, e+1)

	#visualise the final output
	xHAT = gen.forward(Zinit)
	save_image(xHAT.data, join(exDir, 'rec_batch'+str(batchNo)+'.png'), normalize=True, nrow=10)

	return Zinit, recLoss.data.item(), xHAT


if __name__=='__main__':
	opts = get_args()

	#Create new subfolder for saving results and training params
	exDir = join(opts.exDir, 'inversionExperiments_wSTD')
	try:
		os.mkdir(exDir)
	except:
		print('already exists')

	print('Outputs will be saved to:',exDir)
	save_input_args(exDir, opts)

	####### Test data set #######
	IM_SIZE = opts.imSize
	
	print('Prepare data loaders...')
	transform = transforms.Compose([ transforms.ToPILImage(), transforms.Resize((IM_SIZE, IM_SIZE)), \
		transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	if opts.data == 'CELEBA':
		testDataset = CELEBA(root=opts.root, train=False, transform=transform, Ntest=100)  #most models trained with Ntest=1000, but using 100 to prevent memory errors
		gen = GEN(imSize=IM_SIZE, nz=opts.nz, fSize=opts.fSize)
	elif opts.data == 'OMNI':
		print('using Omniglot eval dataset...')
		testDataset = OMNI(root=opts.root, train=False, transform=transform)
		gen = GEN1D(imSize=IM_SIZE, nz=opts.nz, fSize=opts.fSize)
	else:
		testDataset = SHOES(root=opts.root, train=False, transform=transform)
		gen = GEN(imSize=IM_SIZE, nz=opts.nz, fSize=opts.fSize)
	testLoader = torch.utils.data.DataLoader(testDataset, batch_size=opts.batchSize, shuffle=False)
	print('Data loaders ready.')

	print("Test Data Size: ", len(testDataset))

	###### Create model and load parameters #####
	if gen.useCUDA:
		print('Setting cuda device')
		torch.cuda.set_device(opts.gpuNo)
		gen.cuda()
	#gen.load_params(opts.exDir, gpuNo=opts.gpuNo)
	print('params loaded')


	#Find each z individually for each x
	allRec = []
	allX = []
	sumLoss = 0
	for i, data in enumerate(testLoader):
		x, y = prep_data(data, useCUDA=gen.useCUDA)
		z, recLoss, xRec = find_batch_z(gen=gen, x=x, nz=opts.nz, lr=opts.lr, exDir=exDir, maxEpochs=opts.maxEpochs, alpha=opts.alpha, batchNo=i)

		allRec.append(xRec.data)
		allX.append(x.data) #incase the loader shuffles samples

		if opts.oneBatch:
			diff = np.asarray((xRec.data - x.data)**2)
			mseLoss = np.mean(diff, axis=(1,2,3))  # mean over colour channels and pixels
			np.save(join(exDir, 'one_batch_mseLosses_per_sample.npy'), mseLoss)
			meanLoss = np.mean(mseLoss) # mean over samples
			stdLoss = np.std(mseLoss)  #std over samples

			f = open(join(exDir,'one_batch_recError.txt'), 'w')
			f.write('mean loss (one batch) %0.5f' % (meanLoss))
			f.write('std of loss(one batch) %0.5f' % (stdLoss))
			f.write('Test Data (one batch) %d' % np.shape(mseLoss).item())
			f.close()


	allRec = np.concatenate(allRec)
	allX = np.concatenate(allX)
	print('allRec:', np.shape(allRec))
	print('allX:', np.shape(allX))


	mseLoss = np.mean((allRec - allX)**2, axis=(1,2,3))  # mean over colour channels and pixels
	np.save(join(exDir, 'mseLosses_per_sample.npy'), mseLoss)
	meanLoss = np.mean(mseLoss) # mean over samples
	stdLoss = np.std(mseLoss)  #std over samples

	f = open(join(exDir,'recError.txt'), 'w')
	f.write('mean loss %0.5f' % (meanLoss))
	f.write('std of loss %0.5f' % (stdLoss))
	f.write('Test Data %d' % len(testDataset))
	f.close()







