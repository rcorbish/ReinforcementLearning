#!/usr/bin/env python

import time
import random
import gym
import VehicleBall_v0
import model 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def main() :
    learner = Learner() 
    learner.learn() 
    learner.exploit() 


class Learner() :

    BATCH_SIZE = 50
    NUM_STEPS = 100
    DECAY = 0.99
    CONSECUTIVE_FRAME_COUNT=4
    FULLY_RANDOM = 1.0
    

    def __init__( self ) :
        self.env = gym.make( "VehicleBall-v0", render_mode='human' )
        
        self.num_actions = self.env.action_space.n

        # 4 obs + 1 action
        self.numInputs = Learner.CONSECUTIVE_FRAME_COUNT * \
                            self.env.observation_space.shape[0] + 1   

        self.mlp = model.Model( self.numInputs, 1 )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD( self.mlp.parameters(), lr=0.01 )

        self.clear()


    def clear( self ) :
        #  These keep state of the model
        self.rewards = []
        self.observations = []
        self.actions = []
        self.env.reset() 


    def learn( self, num_iterations=10 ) :
        self.clear()
        for it in range(num_iterations) :
            inputs, values, actions = self.collect( self.FULLY_RANDOM, self.BATCH_SIZE )

            for i in range( len(actions) ) :
                inputs[i].append( float(actions[i]) )

            inputs = torch.tensor( inputs )
            values = torch.tensor( values ).unsqueeze(1)
            loss = self.applyLearning( inputs, values )

            print( it, loss )



    def ensure_data_available( self, random_chance, num_steps ) :

        num_needed = num_steps + self.NUM_STEPS -   \
                     len( self.observations ) +     \
                     self.CONSECUTIVE_FRAME_COUNT 

        for step in range( num_needed ) :
            obs, reward, action, done = self.step( random_chance )

            self.observations.append( obs ) 
            self.rewards.append( reward ) 
            self.actions.append( action )

            if done :
                self.env.reset()


    def collect( self, random_chance, num_steps ) :
        self.ensure_data_available( random_chance, num_steps )

        values = [] 
        for ix in range( num_steps ) :
            v = self.discountRewards( self.rewards, ix+self.CONSECUTIVE_FRAME_COUNT-1, self.NUM_STEPS )
            values.append( v ) 

        a = self.actions[ self.CONSECUTIVE_FRAME_COUNT-1:num_steps+self.CONSECUTIVE_FRAME_COUNT-1 ]

        inputs = []
        for i in range( num_steps) :
            o = [] 
            for j in range(i, i+self.CONSECUTIVE_FRAME_COUNT ) :
                o.extend( self.observations[j] )
            inputs.append( o )

        del self.observations[0:num_steps]
        del self.rewards[0:num_steps]
        del self.actions[0:num_steps]

        return inputs, values, a


    def step( self, random_chance ) :
        action = self.decide_action( random_chance ) 
        obs, reward, done, info = self.env.step( action ) 

        return obs, reward, action, done


    def decide_action( self, random_chance=FULLY_RANDOM ) :

        if random.random() < random_chance or len( self.observations ) < self.CONSECUTIVE_FRAME_COUNT :
            return self.env.action_space.sample() 

        frames = []
        for i in range( self.CONSECUTIVE_FRAME_COUNT ) :
            frames.extend( self.observations[i] )

        state = torch.tensor( frames + [ 0.0 ] )
        best_action = 0
        best_value = self.mlp( state ).item()

        for action in range( 1, self.num_actions ) :
            state = torch.tensor( frames + [ float(action) ] )
            state_value = self.mlp( state ).item()
            if state_value > best_value :
                best_action = action
                best_value = state_value

        return best_action


    def exploit(self, iterations=-1) :
        
        random_chance = self.FULLY_RANDOM 
        always = iterations == -1
        it = 0
        while always or iterations>0 :
            it = it + 1
            # with torch.no_grad() :
            self.clear()
            random_chance = random_chance - 0.05
            if random_chance < 0.03 :
                random_chance = 0.03

            input_history = [] 
            value_history = []
            for _ in range(5000) :
                inputs, values, actions = self.collect( random_chance, 1 )
                
                for i in range( len(actions) ) :
                    inputs[i].append( float(actions[i]) )
                # Add new observation to end & 'forget' the oldest observation 
                input_history.extend( inputs )
                value_history.extend( values )

                if len(input_history) > self.BATCH_SIZE :
                    inputs = torch.tensor( input_history )
                    values = torch.tensor( value_history ).unsqueeze(1)
                    loss = self.applyLearning( inputs, values )
                    print( it, loss, random_chance )
                    input_history = [] 
                    value_history = []

            iterations = iterations - (0 if always else 1)


    def applyLearning( self, input, values ) :
        output = self.mlp( input )

        loss = self.criterion( output, values )

        self.optimizer.zero_grad()   # zero the gradient buffers
        loss.backward()     # calc gradients
        self.optimizer.step()    # update weights

        return loss.item() if loss else None


    def discountRewards( self, rewards, offset, length, gamma=0.99 ) :
        value = 0 
        for ix in range( offset+length, offset, -1 ):
            value = value * gamma + rewards[ix-1]

        return value 



if __name__ == "__main__":
    VehicleBall_v0.init()
    main()
 