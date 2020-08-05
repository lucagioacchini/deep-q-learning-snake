# Deep Q-Learning Applied to the Snake Arcade Game
Implementation of a Deep Q-Learning neural network able to automatically play the Snake arcade game.

## Install
Open a terminal and type  
```sudo apt-get install python3-pygame```

## Usage
To train the network and the agent run the ```train.py``` script. To make the 
training faster it is recommended to uncomment ```H=320``` at line 7 of the ```lib.config.py``` file.  
To test the network and the game run the ```test.py``` script. To observe the network generalization provided by the DQL technique it is recommended to uncomment ```H=620``` at line 8 of the ```lib.config.py``` file

## Documentation
To obtain an overview of the Reinforcement learning and the Deep Q-Learning concepts please check the **DOCUMENTATION***.  

The neural network is composed by one input layer with 11 neurons, one output layer with 4 neurons and three dense hidden layer with 150 neurons activated by the relu function.  

The state array representing the network fed as input to the neural network is a one-hot encoded array reporting the following information:
1. 1 if there is an obstacle on the right by the snake POV, 0 otherwise;
2. 1 if there is an obstacle on the left by the snake POV, 0 otherwise;
3. 1 if there is an obstacle forward wrt by the snake POV, 0 otherwise;
4. 1 if there is the food on the right wrt the snake head, 0 otherwise;
5. 1 if there is the food on the left wrt the snake head, 0 otherwise;
6. 1 if there is the food on the top wrt the snake head, 0 otherwise;
7. 1 if there is the food on the bottom wrt the snake head, 0 otherwise;
8. 1 if the snake is gowing down, 0 otherwise;
9. 1 if the snake is gowing up, 0 otherwise;
10. 1 if the snake is gowing right, 0 otherwise;
11. 1 if the snake is gowing left, 0 otherwise.  

## Future works
* Change the training method alternating a one-block snake initialization to encourage the food eating with a more-than-6 blocks snake to prevent the loop formation.  
* Try a convolutional neural network providing 4 subsequent screen captures as input state.

##

(c) 2020, Luca Gioacchini