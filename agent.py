from keras import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
import random 
import math
import numpy as np

GAMMA = .9

class DQN():
    def __init__(self):
        self.model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Dense(50, input_shape=(11,), activation='relu'))
        #model.add(Dropout(.2))
        model.add(Dense(50, activation='relu'))
        #model.add(Dropout(.2))
        #model.add(Dense(150, input_shape=(11,), activation='relu'))
        #model.add(Dropout(.2))
        model.add(Dense(4, activation='softmax'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=.001), metrics=['accuracy'])
        #model.summary()
        return model
    
class ReplayMemory():
    def __init__(self, model):
        self.model = model
        self.capacity = 10000
        self.memory = []
        self.push_count = 0
        self.batch_size = 10
    
    def push(self, experience):
        if len(self.memory) < self.capacity:
            # Start achieving memory from experience
            self.memory.append(experience)
        else:
            # Progressively replace the acquired experience
            # with fresher one
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1
    
    def sample(self):
        # Randomly sample memory
        return random.sample(self.memory, self.batch_size)

    def replay(self, stop):
        if len(self.memory) >= self.batch_size:
            batch = self.sample()
        else:
            batch = self.memory
        
        for state, act, reward, nxt_state in batch:
            nxt_state = np.reshape(nxt_state, (1,11))
            state = np.reshape(state, (1,11))  
            target = reward
            if not stop:
                target = reward + GAMMA * np.amax(self.model.predict(nxt_state)[0])
            target_f = self.model.predict(state)
            target_f[0][np.argmax(act)] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
    
    def exploit(self, state):
        state = np.reshape(state, (1,11))  
        return  np.amax(self.model.predict(state)[0])
            
    


    

class Agent():
    def __init__(self):
        self.model = DQN().model
        self.memory = ReplayMemory(self.model)

    def getState(self, snake, food):
        """
        OB_R, OB_L, OB_F, 
        FOOD_R, FOOD_L, FOOD_U, FOOD_D
        DIR_D, DIR_U, DIR_R, DIR_L
        """
        state = np.zeros(11, dtype=int)

        # Snake goes down
        if snake.dir == 0:
            state[7] = True
            # Obstacle Right
            head = (snake.x[0]-20 , snake.y[0])
            for i in range(1, snake.len):
                if head == (snake.x[i], snake.y[i]) or head[0] < 0: 
                    state[0] = 1
                    break
            # Obstacle Left
            head = (snake.x[0]+20 , snake.y[0])
            for i in range(1, snake.len):
                if head == (snake.x[i], snake.y[i]) or head[0] > 580: 
                    state[1] = 1
                    break
            # Obstacle Forward
            head = (snake.x[0] , snake.y[0]+20)
            for i in range(1, snake.len):
                if head == (snake.x[i], snake.y[i]) or head[1] > 580: 
                    state[2] = 1
                    break
            
        # Snake goes Up
        if snake.dir == 2:
            state[8] = True
            # Obstacle Right
            head = (snake.x[0]+20 , snake.y[0])
            for i in range(1, snake.len):
                if head == (snake.x[i], snake.y[i]) or head[0] > 580: 
                    state[0] = 1
                    break
            # Obstacle Left
            head = (snake.x[0]-20 , snake.y[0])
            for i in range(1, snake.len):
                if head == (snake.x[i], snake.y[i]) or head[0] < 0: 
                    state[1] = 1
                    break
            # Obstacle Forward
            head = (snake.x[0] , snake.y[0]-20)
            for i in range(1, snake.len):
                if head == (snake.x[i], snake.y[i]) or head[1] < 0: 
                    state[2] = 1
                    break
            
        # Snake goes Right
        if snake.dir == 1:
            state[9] = 1
            # Obstacle Right
            head = (snake.x[0] , snake.y[0]+20)
            for i in range(1, snake.len):
                if head == (snake.x[i], snake.y[i]) or head[1] > 580: 
                    state[0] = 1
                    break
            # Obstacle Left
            head = (snake.x[0] , snake.y[0]-20)
            for i in range(1, snake.len):
                if head == (snake.x[i], snake.y[i]) or head[1] < 0: 
                    state[1] = 1
                    break
            # Obstacle Forward
            head = (snake.x[0]+20 , snake.y[0])
            for i in range(1, snake.len):
                if head == (snake.x[i], snake.y[i]) or head[0] > 580: 
                    state[2] = 1
                    break
        
        # Snake goes Left
        if snake.dir == 3:
            state[10] = 1
            # Obstacle Right
            head = (snake.x[0] , snake.y[0]-20)
            for i in range(1, snake.len):
                if head == (snake.x[i], snake.y[i]) or head[1] < 0: 
                    state[0] = 1
                    break
            # Obstacle Left
            head = (snake.x[0] , snake.y[0]+20)
            for i in range(1, snake.len):
                if head == (snake.x[i], snake.y[i]) or head[1] > 580: 
                    state[1] = 1
                    break
            # Obstacle Forward
            head = (snake.x[0]-20 , snake.y[0])
            for i in range(1, snake.len):
                if head == (snake.x[i], snake.y[i]) or head[0] < 0: 
                    state[2] = 1
                    break

        # Food position wrt head            
        state[3] = int(snake.x[0]<food.pos[0])  # Food Right
        state[4] = int(snake.x[0]>food.pos[0])  # Food Left
        state[5] = int(snake.y[0]>food.pos[1])  # Food Up
        state[6] = int(snake.y[0]<food.pos[1])  # Food Down

        return state 
    
    def getEpsilon(self, current_step):
        start = 1
        end = .01
        decay = .05

        eps = end + (start - end)*math.exp(-1*current_step*decay)

        return eps

    def setReward(self, died, ate):
        reward = 0
        if died: reward = -10
        elif ate: reward = 10
        else: reward = 0

        return reward
    