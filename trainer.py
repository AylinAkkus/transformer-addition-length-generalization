__all__ = ['Trainer', 'train_model']


from collections import defaultdict
import torch as t
import torch.optim as optim
import time
import torch.nn.functional as F
import helpers
from helpers import *
from dataclasses import asdict
import os
import wandb
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from config import *
from dataset import *
from model import *
from tokenizer import *

root = helpers.root


class Trainer:
    '''TODO
    ways this stinks:
    - callbacks every k epochs 
    - training on infinite data
    - general abstract class w/o assumption and subclasses w/ more assumptions
    - check out hugging face trainer
    - disentangle optimization step and taking gradients
    - forward compatibility, e.g. batches per step
    '''

    def __init__(self, config : Config, model = None) -> None:
        wandb.init(project = "grokking", config = asdict(config))
        self.model = model if model is not None else Transformer(config, use_cache=False)
        self.model.to(config.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr = config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.98))

        def lr_lambda(step, num_epochs=config.num_epochs, n_warmup = config.n_warmup):
            if step <= n_warmup:
                return min(step / n_warmup, 1)  # Linear warm-up
            elif step >= num_epochs * 0.75:
                return 0.1
            elif step >= num_epochs * 0.5:
                return 0.5
            else:
                return 1.0

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda) # TODO make this a config option
        self.run_name = f"mod_digit_add_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        if config.save_models:
            os.makedirs(root/self.run_name, exist_ok=True)
            os.makedirs(root/self.run_name/'models', exist_ok=True)
        self.data = gen_train_test(config = config)
        self.train = TokenizedDataset(self.data, train=True)
        self.test = TokenizedDataset(self.data, train=False)

        self.metrics_dictionary = defaultdict(dict) # so we can safely call 'update' on keys
        print('training length = ', len(self.train))
        print('testing length = ', len(self.test))
        self.train_losses = []
        self.test_losses = []
        self.config = config

    def save_epoch(self, epoch, save_to_wandb = True):
        ''' precondition! train loss and test losses have been appended to '''
        save_dict = {
            'model': self.model.state_dict(),
            'train_loss': self.train_losses[-1],
            'test_loss': self.test_losses[-1],
            'epoch': epoch,
        }
        if save_to_wandb:
            wandb.log(save_dict)
            print("Saved epoch to wandb")
        if self.config.save_models: 
            t.save(save_dict, f"{root}/{self.run_name}/models/{epoch}.pth")
            print(f"Saved model to {root}/{self.run_name}/models/{epoch}.pth")
        self.metrics_dictionary[epoch].update(save_dict)

    def do_a_training_step(self, epoch: int):
        '''returns train_loss, test_loss'''
        dataloader_train = DataLoader(self.train, batch_size = self.config.batch_size, shuffle = True)
        self.model.train()

        # Train the model for one epoch using mini-batches
        epoch_loss = []
        for batch in dataloader_train:
            self.optimizer.zero_grad()
            train_loss = loss(config = self.config, model = self.model, data = batch)
            train_loss.backward()
            self.optimizer.step()
            epoch_loss.append(train_loss.item())
        self.scheduler.step()

        # Test the model on entire test set
        # We need to use this weird construction because want the test data to be in the same format as the training data
        dataloader_test = DataLoader(self.test, batch_size = len(self.test), shuffle = False)
        self.model.eval()
        for batch in dataloader_test:
            test_loss = loss(config = self.config, model = self.model, data = self.test)

        # Log the train and test losses
        train_loss = t.tensor(epoch_loss).mean() # we mean the train loss over the epoch
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss.item())

        # Print the train and test losses
        if epoch <= 2*self.config.n_warmup:
            print(f'Epoch {epoch}, train loss {t.log(train_loss).item():.4f}, test loss {t.log(test_loss).item():.4f}')
        elif epoch % 10 == 0:
            print(f'Epoch {epoch}, train loss {t.log(train_loss).item():.4f}, test loss {t.log(test_loss).item():.4f}')

        return train_loss, test_loss

    def initial_save_if_appropriate(self):
        """
        Save the model, config and entire data at the start of training
        """
        if self.config.save_models:
            os.makedirs(root/self.run_name, exist_ok=True)

            # Save model
            save_dict = {'model': self.model.state_dict()}
            t.save(save_dict, root/self.run_name/'init.pth')
            
            # Save the config
            config_json = self.config.serialize()
            with open(f"{root}/{self.run_name}/config.json", 'w') as f:
                f.write(config_json)

            # Save entire data as csv
            self.data.to_csv(f"{root}/{self.run_name}/data.csv")



    def post_training_save(self, save_optimizer_and_scheduler = True, log_to_wandb = True):
        if self.config.save_models:
            os.makedirs(root/self.run_name, exist_ok=True)
        save_dict = {
            'model': self.model.state_dict(),
            'train_loss': self.train_losses[-1],
            'test_loss': self.test_losses[-1],
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'epoch': self.config.num_epochs,
        }
        if save_optimizer_and_scheduler:
            save_dict['optimizer'] = self.optimizer.state_dict()
            save_dict['scheduler'] = self.scheduler.state_dict()
        if log_to_wandb:
            wandb.log(save_dict)
        t.save(save_dict, root/self.run_name/f"final.pth")
        print(f"Saved model to {root/self.run_name/f'final.pth'}")
        self.metrics_dictionary[save_dict['epoch']].update(save_dict)


    def take_metrics(self, train, epoch):
        """
        with t.inference_mode():
            def sum_sq_weights():
                # TODO refactor- taken from app code
                row = []
                for name, param in self.model.named_parameters():
                    row.append(param.pow(2).sum().item())
                return row

            print('taking metrics')

            all_data = t.tensor([(i, j, self.config.p) for i in range(self.config.p) for j in range(self.config.p)]).to(self.config.device)
            # TODO calculate key freqs is the most expensive part of this
            key_freqs = calculate_key_freqs(config = self.config, model = self.model, all_data = all_data) 
            logits = self.model(all_data)[:, -1, :-1] # TODO i think this is equivalent to what's in the new paper?
            fourier_basis = make_fourier_basis(config = self.config)
            is_train, is_test = self.config.is_train_is_test(train = train)
            labels = t.tensor([self.config.fn(i, j) for i, j, _ in all_data]).to(self.config.device)

            metrics = {
                'epoch': epoch, 
                'trig_loss': calculate_trig_loss(config = self.config,
                    model = self.model,
                    train = train,
                    key_freqs = key_freqs,
                    is_test=is_test,
                    is_train=is_train,
                    labels=labels,
                    logits = logits,
                    fourier_basis=fourier_basis, 
                    all_data=all_data),
                'sum_of_squared_weights': sum_sq_weights(),
                'excluded_loss': calculate_excluded_loss(
                    logits = logits,
                    key_freqs = key_freqs,
                    fourier_basis = fourier_basis,
                    is_train=is_train,
                    config = self.config,
                    is_test = is_test,
                    labels=labels),
                'coefficients': calculate_coefficients(p = self.config.p, logits = logits, fourier_basis = fourier_basis, key_freqs = key_freqs, device = self.config.device),
            }
            wandb.log(metrics)
            print("Logged metrics to wandb")
            self.metrics_dictionary[epoch].update(metrics)
        """
        # We store the frequencies for the metrics in this epoch in a dictionary
        metrics = {}

        def check_all_digits(pred, ground_truth, data):
            """
            Updates the frequency for the all_digits metric
            """
            if ground_truth == pred:
                if data.equals(train):
                    try:
                        metrics['train_accuracy_total'] += 1
                    except:
                        metrics['train_accuracy_total'] = 1
                else:
                    try:
                        metrics['test_accuracy_total'] += 1
                    except:
                        metrics['test_accuracy_total'] = 1

        def check_individual_digits(pred, ground_truth, data):
            """
            Updates the frequency for the individual digits metric
            """
            # Fill shorter number with zeros
            max_len = max(len(str(pred)), len(str(ground_truth)))
            pred_str = f"{pred:0{max_len}d}"
            ground_truth_str = f"{ground_truth:0{max_len}d}"
            
            assert len(pred_str) == len(ground_truth_str)

            for i in range(len(pred_str)):
                if pred_str[-(i+1)] == ground_truth_str[-(i+1)]:
                    if data.equals(train):
                        try:
                            metrics[f'train_accuracy_digit_{i}'] += 1
                        except:
                            metrics[f'train_accuracy_digit_{i}'] = 1
                    else:
                        try:
                            metrics[f'test_accuracy_digit_{i}'] += 1
                        except:
                            metrics[f'test_accuracy_digit_{i}'] = 1

        # We use the df instead of the dataset class for this
        # because our generate greedy function is not batched
        train = self.data[self.data["is_train"]==True]
        test = self.data[self.data["is_train"]==False]

        # Calculate the accuracy (digits and overall) for train and test
        for data in [train, test]:
            for i, row in data.iterrows():

                # Get the ground truth and prediction
                ground_truth = int(row["result"])
                try: 
                    pred = int(self.model.generate_greedy(row["tokenized"])[0])
                except:
                    print("Prediction can not be cast into int")
                    print("input", row["input_str"])
                    print("Prediction", pred)

                # Check prediction and ground truth for each digit and overall
                check_all_digits(pred, ground_truth, data)
                check_individual_digits(pred, ground_truth, data)

            # Calculate the accuracy from the frequencies
            for key in metrics.keys():
                if "train" in key:
                    metrics[key] = metrics[key] / len(train)
                else:
                    metrics[key] = metrics[key] / len(test)

        # Log the metrics dictionary accross all epochs
        self.metrics_dictionary[epoch].update(metrics)
        #print("metrics", metrics)
        

      

def train_model(config: Config):
    world = Trainer(config = config)
    print(f'Run name {world.run_name}')
    world.initial_save_if_appropriate()

    for epoch in range(config.num_epochs):
        t0 = time.time()
        train_loss, test_loss = world.do_a_training_step(epoch)
        #print(f"Epoch {epoch} took {time.time() - t0:.2f} seconds")
        if test_loss.item() < config.stopping_thresh:
            break
        if config.is_it_time_to_save(epoch = epoch):
            # TODO this also used to do a check about test loss- pretty sure not necessary
            world.save_epoch(epoch = epoch)
        if config.is_it_time_to_take_metrics(epoch = epoch):
            world.take_metrics(epoch = epoch, train = world.train)

    world.post_training_save(save_optimizer_and_scheduler=True)
    helpers.lines([world.train_losses, world.test_losses], labels=['train', 'test'], log_y=True)
    return world # to export the dictionary with the training metrics
        

if __name__ == '__main__':
    config = Config()
    trainer = Trainer(config)
    trainer.take_metrics(train = True, epoch = 0)