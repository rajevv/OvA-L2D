import torch.nn as nn
import torch
import numpy as np




class Criterion(object):
	def __init__(self):
		pass
		
	def softmax(self, outputs, m, labels, m2, n_classes):
		'''
		The L_{CE} loss implementation for CIFAR
		----
		outputs: network outputs
		m: cost of deferring to expert cost of classifier predicting (alpha* I_{m\neq y} + I_{m =y})
		labels: target
		m2:  cost of classifier predicting (alpha* I_{m\neq y} + I_{m =y})
		n_classes: number of classes
		'''
		batch_size = outputs.size()[0]  # batch_size
		rc = [n_classes] * batch_size
		outputs = -m * torch.log2(outputs[range(batch_size), rc]) - m2 * torch.log2(
			outputs[range(batch_size), labels])  
		return torch.sum(outputs) / batch_size