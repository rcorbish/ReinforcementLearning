# -*- coding: utf-8 -*-

import time
import random
import gym
import VehicleBall_v0
import model 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def main( model_file, num_iterations, learning_rate ) :
    learner = Learner() 

    learner.learn( 1000 ) 
    learner.exploit() 


class Learner() :

    BATCH_SIZE = 50
    NUM_STEPS = 100
    DECAY = 0.99
    CONSECUTIVE_FRAME_COUNT=4
    FULLY_RANDOM = 1.0
    

    def __init__( self, model_file, hidden_sizes, num_iterations, learning_rate ) :
        self.env = gym.make( "VehicleBall-v0", render_mode='human' )
        
        self.num_actions = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 4 obs + 1 action
        self.numInputs = Learner.CONSECUTIVE_FRAME_COUNT * \
                            self.env.observation_space.shape[0] + 1   

        self.mlp = model.Model( self.numInputs, 1, hidden_sizes, model_file ).to( self.device )
        self.optimizer = optim.SGD( self.mlp.parameters(), lr=learning_rate )

        self.model_file = model_file
        self.num_iterations = num_iterations
        
        self.clear()


    def clear( self ) :
        obs = self.env.reset() 

        #  Reset state of the model
        self.rewards = []
        self.observations = []
        self.actions = []
        self.dones = []

        for _ in range( self.CONSECUTIVE_FRAME_COUNT ) :
            self.observations.append( obs )



    def learn( self ) :
        self.clear()

        for it in range( self.num_iterations ) :
            x, y, actions = self.collect( self.FULLY_RANDOM, self.BATCH_SIZE )

            if len( x ) == 0 :
                break
            for i in range( len(x) ) :
                x[i].append( float(actions[i]) )

            x = torch.tensor( x, device=self.device )
            y = torch.tensor( y, device=self.device ).unsqueeze(1)
            loss = self.applyLearning( x, y )

            print( it, loss )



    def step( self, random_chance, num_steps=1 ) :

        num_needed = num_steps + self.NUM_STEPS - len( self.actions )

        for start in range( num_needed ) :
            action = self.decide_action( start, random_chance ) 
            obs, reward, done, _ = self.env.step( action ) 

            # for BCE loss we need inputs normalized 0 .. 1
            obs = list( map( lambda o : o + 1.0 / 2.0, obs ) )

            # previous_obs | action => obs | reward
            self.observations.append( obs ) 
            self.rewards.append( reward ) 
            self.actions.append( action )
            self.dones.append( done )



    def collect( self, random_chance, num_steps=1 ) :
        self.step( random_chance, num_steps )

        values = [] 
        for ix in range( num_steps ) :
            # discount rewards from resulting state N steps in the future
            v = self.discountRewards( self.rewards, ix, self.NUM_STEPS )
            values.append( v ) 

        inputs = []
        for i in range( num_steps) :
            o = [] 
            for j in range( i, i+self.CONSECUTIVE_FRAME_COUNT ) :
                o.extend( self.observations[j] )
            inputs.append( o )

        actions = self.actions[ 0:num_steps ]

        try :
            n = self.dones.index( True )
        except ValueError :
            n = num_steps 

        del self.observations[0:num_steps]
        del self.rewards[0:num_steps]
        del self.actions[0:num_steps]
        del self.dones[0:num_steps]

        return inputs[0:n], values[0:n], actions[0:n]



    def decide_action( self, start, random_chance=FULLY_RANDOM ) :
        if random.random() < random_chance :
            return self.env.action_space.sample() 

        frames = []
        for i in range( self.CONSECUTIVE_FRAME_COUNT ) :
            frames.extend( self.observations[i + start] )

        best_action = 0
        with torch.no_grad() :
            state = torch.tensor( frames + [ 0.0 ], dtype=torch.float,device=self.device )
            yhat, recon_x, mu, logvar = self.mlp( state )
            best_value = yhat.item() 
            
            for action in range( 1, self.num_actions ) :
                state[ len(frames) ] = float(action)
                yhat, _, _, _ = self.mlp( state )
                state_value = yhat.item()

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
            for stp in range(5000) :
                inputs, values, actions = self.collect( random_chance )
                if len(inputs) == 0 :
                    break # we're done !

                inputs[0].append( float(actions[0]) )
                # Add new observation to end & 'forget' the oldest observation 
                input_history.extend( inputs )
                value_history.extend( values )

                if len(input_history) > self.BATCH_SIZE :
                    inputs = torch.tensor( input_history, device=self.device )
                    values = torch.tensor( value_history, device=self.device ).unsqueeze(1)
                    loss = self.applyLearning( inputs, values )
                    print( it, loss, random_chance )
                    input_history = [] 
                    value_history = []

                if ( stp & 63 ) == 0 :
                    self.photo() 

            iterations = iterations - (0 if always else 1)
            self.mlp.save()



    def applyLearning( self, x, y ) :

        yhat, recon_x, mu, logvar = self.mlp( x )

        loss = self.mlp.loss( yhat, y, recon_x, x, mu, logvar )

        self.optimizer.zero_grad()  # zero the gradient buffers
        loss.backward()             # calc gradients
        self.optimizer.step()       # update weights

        return loss.item()


    def photo( self ) :
        self.env.render() 


    def discountRewards( self, rewards, offset, length, gamma=0.99 ) :
        value = 0 
        for ix in range( offset+length, offset, -1 ):
            if self.dones[ix] :
                value = 0 
            value = value * gamma + rewards[ix]

        return value 
 