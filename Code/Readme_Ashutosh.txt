
Main function have three different mode for training, testing on public set and for prediction

1) For training -> Change the mode to "train" in Configure.py
2) For testing  -> Change the mode to "test" in Configure.py
3) For prediction  -> Change the mode to "predict" in Configure.py

After configuring mode from configure file, look if you want to change other configurations.
Other parameters that can be changed from Configure file are:
1) Max Epochs
2) Data Directory path
3) Model Directory path to store models
4) Private directory path to read/store prediction input/output
5) Batch Size
6) Learning rate
7) Weight decay
8) Model saving internal
9) Validation List

Once all the parameters and mode is set, run the main file.