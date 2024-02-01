### YOUR CODE HERE
import torch
import torch.nn as nn
import os, time
import numpy as np
from tqdm import tqdm
from Network import DualPathNetwork
from ImageUtils import parse_record, preprocess_image
import pandas as pd

"""This script defines the training, validation and testing process.
"""

class CifarDL(nn.Module):
    def __init__(self, config):
        super(CifarDL, self).__init__()
        self.config = config
        self.network = DualPathNetwork(config)
        ### YOUR CODE HERE
        # define cross entropy loss and optimizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Used device", self.device)
        self.network.to(self.device)
        self.loss_fun = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(),lr=self.config.learning_rate,weight_decay=self.config.weight_decay,momentum=0.9)
   
    
    def train(self, x_train, y_train, max_epoch):
        self.network.train()
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.config.batch_size

        

        print('### Training... ###')
        for epoch in range(1, max_epoch+1):
            start_time = time.time()
            # Shuffle
            #np.random.seed(epoch)
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            ### YOUR CODE HERE
            # Set the learning rate for this epoch
            # Usage example: divide the initial learning rate by 10 after several epochs
            
            if (epoch%60==0 or epoch%90==0 or epoch%140==0):
            	self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr']/15.
            ### YOUR CODE HERE
            
            for i in range(num_batches):
                ### YOUR CODE HERE
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                # Don't forget L2 weight decay
                
                X_batch = curr_x_train[i*self.config.batch_size:min((i+1)*self.config.batch_size,curr_x_train.shape[0])]
                y_batch = curr_y_train[i*self.config.batch_size:min((i+1)*self.config.batch_size,curr_y_train.shape[0])]
                X_batch = np.array(list(map(lambda x: parse_record(x,True),X_batch))) 
                X_batch = torch.tensor(X_batch,device=self.device, dtype=torch.float)
                
                prediction = self.network.forward(X_batch)
                y_batch = torch.tensor(y_batch,device=self.device)
                y_batch = y_batch.to(torch.long)
                loss = self.loss_fun(prediction,y_batch)
                ### YOUR CODE HERE
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)
                print('\n')
            print('\n')
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))

            if epoch % self.config.save_interval == 0:
                self.save(epoch)


    def evaluate(self, x, y, valid_list):
        self.network.eval()
        print('### Test or Validation ###')
        for num in valid_list:
            valid_file = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(num))
            self.load(valid_file)

            preds = []
            with torch.no_grad():
                for i in tqdm(range(x.shape[0])):
                    ### YOUR CODE HERE
                    x_processed = np.array(list(map(lambda x: parse_record(x,False),x[i:i+1])))
                    x_processed = torch.tensor(x_processed,device=self.device,dtype=torch.float)
                    preds.append(torch.argmax(self.network.forward(x_processed),axis=1))
                    ### END CODE HERE

            y = torch.tensor(y)
            preds = torch.tensor(preds)
            print('Test accuracy: {:.4f}'.format(torch.sum(preds==y)/y.shape[0]))
    
    def save(self, epoch):
        checkpoint_path = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(epoch))
        os.makedirs(self.config.modeldir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, name):
        ckpt = torch.load(name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(name))
    
    def predict_prob(self, x, num):
        
        best_model = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(num))
        self.load(best_model)
        self.network.eval()

        preds = []
        preds_max=[]
        with torch.no_grad():
            for i in tqdm(range(x.shape[0])):

                ### YOUR CODE HERE
                #processing the x before test pred
                test_x_proc = np.array(list(map(lambda x: parse_record(x,False),x[i:i+1])))
                test_x_proc = torch.tensor(test_x_proc,device=self.device,dtype=torch.float)
                pred = self.network.forward(test_x_proc)
                temp=torch.nn.functional.softmax(pred).cpu().detach().numpy().reshape((10))
                preds.append(temp)
                
                preds_max.append(np.argmax(temp))

        return preds
