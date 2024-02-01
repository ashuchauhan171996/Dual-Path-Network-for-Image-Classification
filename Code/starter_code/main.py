  ### YOUR CODE HERE
import torch
import os, argparse
import numpy as np
from Model import CifarDL
from DataLoader import load_data, train_valid_split, load_testing_images
import Configure
from ImageUtils import visualize

if __name__ == '__main__':
	
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  config = Configure.configure()
  print('lets go')
  
  print("--- Preparing Data ---")
  
  data_dir = config.datadir
  # Load train and test data
  x_train, y_train, x_test, y_test = load_data(data_dir)
  # Split train data into train_new and valid
  x_train_new, y_train_new, x_valid, y_valid = train_valid_split(x_train, y_train)

  model = CifarDL(config).cuda()
  #print(model)

  if config.mode == 'train':
    
    print("Training data X and Y shape for training: x_train_new:", x_train_new.shape, ", y_train_new:", y_train_new.shape)
    model.train(x_train_new, y_train_new, config.max_epoch)
    
    model.evaluate(x_valid, y_valid, config.valid_list)
    
    # print("\n")
    # print("Training data on complete training set with hyperparameter obtained in validation")
    # model.train(x_train, y_train, 190)
    

  elif config.mode == 'test':
    # Testing on public testing dataset
    
    print("\n")
    print("\nTesting model on test set")
    model.evaluate(x_test, y_test, [190])

  elif config.mode == 'predict':
    print('Loading private testing dataset')
    x_private_test = load_testing_images(config.privatedir)
    # visualizing the first testing image to check your image shape
    visualize(x_private_test[0], 'test.png')
    # Predicting and storing results on private testing dataset 
    predictions = model.predict_prob(x_private_test, 190)
    np.save(config.privatedir+'predictions.npy', predictions)
