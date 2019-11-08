import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

import utils
from model import DeepARLSTM
from dataloader import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




def evaluate(model, loss_fn, test_loader, test_predict_start, device, sample=True):

    model.eval()
    with torch.no_grad():
      plot_batch = np.random.randint(len(test_loader)-1)

      summary_metric = {}
      raw_metrics = utils.init_metrics(sample=sample)

      # Test_loader:
      # test_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
      # id_batch ([batch_size]): one integer denoting the time series id;
      # v ([batch_size, 2]): scaling factor for each window;
      # labels ([batch_size, train_window]): z_{1:T}.
      for i, (test_batch, id_batch, v, labels) in enumerate(tqdm(test_loader)):
          # test_batch: (sequence, test_batch, feature)
          # id_batch: (1, test_batch)
          # v_batch: (test_batch, 2)
          # labels: (batch, sequence)

          test_batch = test_batch.permute(1, 0, 2).to(torch.float32).to(device)
          id_batch = id_batch.unsqueeze(0).to(device)
          v_batch = v.to(torch.float32).to(device)
          labels = labels.to(torch.float32).to(device)
          batch_size = test_batch.shape[1]
          input_mu = torch.zeros(batch_size, test_predict_start, device=device) # scaled
          input_sigma = torch.zeros(batch_size, test_predict_start, device=device) # scaled
          hidden = model.init_hidden(batch_size)
          cell = model.init_cell(batch_size)

          for t in range(test_predict_start):
              # if z_t is missing, replace it by output mu from the last time step
              zero_index = (test_batch[t,:,0] == 0)
              if t > 0 and torch.sum(zero_index) > 0:
                  test_batch[t,zero_index,0] = mu[zero_index]

              mu, sigma, hidden, cell = model(test_batch[t].unsqueeze(0), id_batch, hidden, cell)
              input_mu[:,t] = v_batch[:, 0] * mu + v_batch[:, 1]
              input_sigma[:,t] = v_batch[:, 0] * sigma

          if sample:
              samples, sample_mu, sample_sigma = model.test(test_batch, v_batch, id_batch, hidden, cell, 200, 168, 24, sampling=True)
              raw_metrics = utils.update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels, test_predict_start, samples, relative = True)
          else:
              sample_mu, sample_sigma = model.test(test_batch, v_batch, id_batch, hidden, cell)
              raw_metrics = utils.update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels, test_predict_start, relative = True)

          if i == plot_batch:
              if sample:
                  sample_metrics = utils.get_metrics(sample_mu, labels, test_predict_start, samples, relative = True)
              else:
                  sample_metrics = utils.get_metrics(sample_mu, labels, test_predict_start, relative = True)
              # select 10 from samples with highest error and 10 from the rest
              top_10_nd_sample = (-sample_metrics['ND']).argsort()[:batch_size // 10]  # hard coded to be 10
              chosen = set(top_10_nd_sample.tolist())
              all_samples = set(range(batch_size))
              not_chosen = np.asarray(list(all_samples - chosen))
              if batch_size < 100: # make sure there are enough unique samples to choose top 10 from
                  random_sample_10 = np.random.choice(top_10_nd_sample, size=10, replace=True)
              else:
                  random_sample_10 = np.random.choice(top_10_nd_sample, size=10, replace=False)
              if batch_size < 12: # make sure there are enough unique samples to choose bottom 90 from
                  random_sample_90 = np.random.choice(not_chosen, size=10, replace=True)
              else:
                  random_sample_90 = np.random.choice(not_chosen, size=10, replace=False)
              combined_sample = np.concatenate((random_sample_10, random_sample_90))

              label_plot = labels[combined_sample].data.cpu().numpy()
              predict_mu = sample_mu[combined_sample].data.cpu().numpy()
              predict_sigma = sample_sigma[combined_sample].data.cpu().numpy()
              plot_mu = np.concatenate((input_mu[combined_sample].data.cpu().numpy(), predict_mu), axis=1)
              plot_sigma = np.concatenate((input_sigma[combined_sample].data.cpu().numpy(), predict_sigma), axis=1)
              plot_metrics = {_k: _v[combined_sample] for _k, _v in sample_metrics.items()}
              # plot_eight_windows(params.plot_dir, plot_mu, plot_sigma, label_plot, params.test_window, params.test_predict_start, plot_num, plot_metrics, sample)

      summary_metric = utils.final_metrics(raw_metrics, sampling=sample)
      metrics_string = '; '.join('{}: {:05.3f}'.format(k, v) for k, v in summary_metric.items())
    return summary_metric
