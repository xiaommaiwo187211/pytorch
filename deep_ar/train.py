import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from torch.nn.utils import clip_grad_norm_

from time import time
import numpy as np

from dataloader import TimeSeriesDataSet, WeightedSampler
from deepar_model import *



def train_epoch(model, dataloader, optimizer, loss_func, device):
    model.train()

    loss_dict = {'loss': 0, 'mae_loss': 0}
    total_cnt = 0
    for i, (inputs_batch, individuals_batch, targets_batch, _) in enumerate(dataloader):
        # inputs_batch: (sequence, batch, feature)
        # individuals_batch: (batch)
        # targets_batch: (sequence, batch)

        optimizer.zero_grad()

        inputs_batch = inputs_batch.float().to(device).permute(1, 0, 2)
        individuals_batch = individuals_batch.unsqueeze(0).long().to(device)
        targets_batch = targets_batch.float().to(device).permute(1, 0)
        sequence, batch_size, _ = inputs_batch.size()

        loss = torch.zeros(1, device=device, requires_grad=False)
        mae_loss = torch.zeros(1, device=device, requires_grad=False)
        hidden, cell = model.init_hidden_cell(batch_size)
        for t in range(sequence):
            zero_index = (inputs_batch[t, :, 0] == 0)
            if t > 0 and zero_index.sum() > 0:
                inputs_batch[t, zero_index, 0] = mu[zero_index]
            mu, sigma, hidden, cell = model(individuals_batch, inputs_batch[t:t+1, :, :].clone(), hidden, cell)
            loss += loss_func(mu, sigma, targets_batch[t])
            mae_loss += mae(mu, targets_batch[t])

        loss.backward()
        clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        loss_dict['loss'] += loss.item() * batch_size
        loss_dict['mae_loss'] += mae_loss.item() * batch_size
        total_cnt += batch_size

    loss_dict['mae_loss'] /= total_cnt * sequence
    loss_dict['loss'] /= total_cnt * sequence
    return loss_dict



def evaluate_epoch(model, decoder, dataloader, loss_func, device, sampling_times=10, sampling=True):
    model.eval()

    input_seq_len, output_seq_len = decoder.input_seq_len, decoder.output_seq_len
    with torch.no_grad():
        loss_dict = {'loss': 0, 'mae_loss': 0}
        total_cnt = 0
        for i, (inputs_batch, individuals_batch, targets_batch, means_batch) in enumerate(dataloader):

            inputs_batch = inputs_batch.float().to(device).permute(1, 0, 2)
            individuals_batch = individuals_batch.unsqueeze(0).long().to(device)
            targets_batch = targets_batch.float().to(device).permute(1, 0)
            means_batch = means_batch.float().to(device)
            sequence, batch_size, _ = inputs_batch.size()

            hidden, cell = model.init_hidden_cell(batch_size)
            for t in range(input_seq_len):
                zero_index = (inputs_batch[t, :, 0] == 0)
                if t > 0 and zero_index.sum() > 0:
                    inputs_batch[t, zero_index, 0] = mu[zero_index]
                mu, sigma, hidden, cell = model(individuals_batch, inputs_batch[t:t+1, :, :].clone(), hidden, cell)

            # pred_samples: (sample_times, output_sequence, batch)
            # mu_sample: (output_sequence, batch)
            # sigma_sample: (output_sequence, batch)
            if sampling:
                pred_samples, mu_sample, sigma_sample = decoder.decoder_sampling(model, individuals_batch, inputs_batch, hidden, cell, means_batch, sampling_times)
            else:
                mu_sample, sigma_sample = decoder.decoder(model, individuals_batch, inputs_batch, hidden, cell, means_batch)

            mae_loss = torch.sum([mae(mu_sample[b], targets_batch[b, -output_seq_len:]) for b in range(batch_size)])
            loss = torch.sum([loss_func(mu_sample[t], sigma_sample[t], targets_batch[input_seq_len + t]) for t in range(output_seq_len)])
            loss_dict['mae_loss'] += mae_loss.item() * output_seq_len
            loss_dict['loss'] += loss.item() * batch_size
            total_cnt += batch_size

    loss_dict['mae_loss'] /= total_cnt * output_seq_len
    loss_dict['loss'] /= total_cnt * output_seq_len
    return loss_dict



def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



def set_seed():
    SEED = 12
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True




if __name__ == '__main__':
    DATA_PATH = './'
    EPOCH_NUM = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    INDIVIDUAL_SIZE  = 370
    EMBEDDING_SIZE = 20
    INPUTS_SIZE = 5
    HIDDEN_SIZE = 40
    LAYER_NUM = 3
    DROPOUT = 0.1
    INPUT_SEQ_LEN = 168 # encoder length is 7 days
    OUTPUT_SEQ_LEN = 24 # decoder length is 1 day
    SAMPLING_TIMES = 10
    SAMPLING = False

    set_seed()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_set = TimeSeriesDataSet(DATA_PATH, mode='train')
    test_set = TimeSeriesDataSet(DATA_PATH, mode='test')
    # train set should use weighted sampler while test set should use random sampler
    train_sampler = WeightedSampler(DATA_PATH, mode='train')
    test_sampler = RandomSampler(test_set)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, sampler=test_sampler, num_workers=4)

    deepar = DeepARLSTM(INDIVIDUAL_SIZE, EMBEDDING_SIZE, INPUTS_SIZE, HIDDEN_SIZE, LAYER_NUM, DROPOUT, device).to(device)
    optimizer = optim.Adam(deepar.parameters(), lr=LEARNING_RATE)
    loss_func = negative_log_likelihood
    decoder = DeepARDecoder(INPUT_SEQ_LEN, OUTPUT_SEQ_LEN, device)


    min_val_loss = float('inf')
    for epoch in range(EPOCH_NUM):

        start_time = time()

        train_loss_dict = train_epoch(deepar, train_loader, optimizer, loss_func, device)
        train_loss, train_mae_loss = train_loss_dict['loss'], train_loss_dict['mae_loss']

        val_loss_dict = evaluate_epoch(deepar, decoder, test_loader, loss_func, device, SAMPLING_TIMES, SAMPLING)
        val_loss, val_mae_loss = val_loss_dict['loss'], val_loss_dict['mae_loss']

        end_time = time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print('Epoch: %s | Time: %sm %ss' % (str(epoch + 1).zfill(2), epoch_mins, epoch_secs))
        print('\tTrain Loss: %.3f | Val Loss: %.3f | Val MAE Loss: %.3f' % (train_loss, val_loss, val_mae_loss))

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(deepar.state_dict(), 'model/deepar_model.pt')
            print()
            print('model saved with validation loss', val_loss)
            print()