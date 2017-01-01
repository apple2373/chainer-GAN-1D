#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author: Satoshi Tsutsui

This is the very naive implementation of GAN (https://arxiv.org/abs/1406.2661) to learn one dimensional gaussian distribution. 

I consulted on these resources:
http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
https://github.com/AYLIEN/gan-intro/blob/master/gan.py
https://gist.github.com/Newmu/4ee0a712454480df5ee3
'''

import argparse
import numpy as np
from matplotlib import pyplot as plt
import json
from scipy.stats import gaussian_kde
import time

import sys
import os
#os.environ["CHAINER_TYPE_CHECK"] = "0" #to disable type check. 
import chainer 

from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

#Parse arguments
parser = argparse.ArgumentParser()
# parser.add_argument("-g", "--gpu",default=-1, type=int, help=u"GPU ID.CPU is -1")
parser.add_argument("--iter",default=1000, type=int, help=u"# of interations")
parser.add_argument("--ksteps",default=2, type=int, help=u"# of ksteps")
parser.add_argument("--hidden",default=64, type=int, help=u"mini batchsize")
parser.add_argument("--batch",default=128, type=int, help=u"mini batchsize")
args = parser.parse_args()

class DataDistribution(object):
	#taken from https://github.com/AYLIEN/gan-intro/blob/master/gan.py
    def __init__(self):
        self.mu = 5
        self.sigma = 1

    def sample(self, N):
    	#N is batch size 
        samples = np.random.normal(self.mu, self.sigma, N)
        return np.expand_dims(samples, axis=1).astype(np.float32)

class GeneratorDistribution(object):
	#random noize Z
	#taken from https://github.com/AYLIEN/gan-intro/blob/master/gan.py
    def __init__(self, range):
        self.range = range

    def sample(self, N):
    	#N is batch size 
    	samples=np.linspace(-self.range, self.range, N) + np.random.random(N) * 0.01
        return np.expand_dims(samples, axis=1).astype(np.float32)

class Generator(Chain):
     def __init__(self, n_units, n_out=1):
     	#taken from chainer tutorial
         super(Generator, self).__init__(
             # the size of the inputs to each layer will be inferred
             l1=L.Linear(None, n_units),  # n_in -> n_units
             l2=L.Linear(None, n_units),  # n_units -> n_units
             l3=L.Linear(None, n_out),    # n_units -> n_out
         )

     def __call__(self, x):
         h1 = F.leaky_relu(self.l1(x))
         h2 = F.leaky_relu(self.l2(h1))
         y = self.l3(h2)
         return y

class Discriminator(Chain):
     def __init__(self, n_units, n_out=1):
     	#taken from chainer tutorial
         super(Discriminator, self).__init__(
             # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_units),  # n_units -> n_units
            l3=L.Linear(None, n_units),  # n_units -> n_units
            l4=L.Linear(None, n_out),    # n_units -> n_out
         )

     def __call__(self, x,test=False):
		h1 = F.elu(self.l1(x))
		h2 = F.elu(self.l2(h1))
		h3 = F.elu(self.l3(h2))
		y = F.sigmoid(self.l4(h3))
		return y

def gaussian_likelihood(X, u=0., s=1.):
    return (1./(s*np.sqrt(2*np.pi)))*np.exp(-(((X - u)**2)/(2*s**2)))

def vis(i,xs,ps):
    z = gen.sample(500)
    gs = G(z).data.flatten()
    preal = D(xs.reshape(-1, 1),test=True).data.flatten()
    kde = gaussian_kde(gs)

    plt.clf()
    plt.plot(xs, ps, '--', lw=2)
    plt.plot(xs, kde(xs), lw=2)
    plt.plot(xs, preal, lw=2)
    plt.xlim([data.mu-4*data.sigma, data.mu+4*data.sigma])
    plt.ylim([0., 1.])
    plt.ylabel('Prob')
    plt.xlabel('x')
    plt.legend(['P(data)', 'G(z)', 'D(x)'])
    plt.title('GAN learning guassian iter=%d'%i)
    fig.canvas.draw()
    plt.show(block=False)

fig = plt.figure()

data=DataDistribution()
xs = np.linspace(data.mu-4*data.sigma, data.mu+4*data.sigma, 200).astype('float32')
ps = gaussian_likelihood(xs, data.mu,data.sigma)

gen=GeneratorDistribution(range=8)

G=Generator(args.hidden)
D=Discriminator(args.hidden)

#setup optimizer
optimizerG = optimizers.Adam(alpha=0.001)
optimizerG.setup(G)
optimizerG.add_hook(chainer.optimizer.WeightDecay(0.0005))
optimizerD = optimizers.Adam(alpha=0.001)
optimizerD.setup(D)
optimizerD.add_hook(chainer.optimizer.WeightDecay(0.0005))

batch_size=args.batch
for i in xrange(args.iter):
	for k in xrange(args.ksteps):
		optimizerD.zero_grads()
		z_batch=gen.sample(batch_size)
		x_batch=data.sample(batch_size)
		#符号をチェック
		loss_d=F.sum(-F.log(D(x_batch))-F.log(np.ones([batch_size,1])-D(G(z_batch))) )/batch_size
		loss_d.backward()

		optimizerD.update()

	optimizerG.zero_grads()
	z_batch=gen.sample(batch_size)
	x_batch=data.sample(batch_size)
	#loss_g=F.sum( F.log(np.ones([batch_size,1])-D(G(z_batch))) )/batch_size
	loss_g=-F.sum( F.log(D(G(z_batch))) )/batch_size
	loss_g.backward()
	optimizerG.update()

	if i!=0 and i%100==0:
		optimizerD.alpha/=2
		optimizerG.alpha/=2

	print loss_d.data,loss_g.data
	vis(i,xs,ps)






