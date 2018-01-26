import numpy as np

"""
Contains the definition of the agent that will run in an
environment.
"""
ACT_UP    = 1
ACT_DOWN  = 2
ACT_LEFT  = 3
ACT_RIGHT = 4
ACT_TORCH_UP    = 5
ACT_TORCH_DOWN  = 6
ACT_TORCH_LEFT  = 7
ACT_TORCH_RIGHT = 8

class QAgent:
    def __init__(self):
        """Init a new agent.
        """
        self.epsilon = 0.1
        self.gamma = 0.9999
        self.alpha = 0.1

        self.Q = np.ones((100,100,2,2,5,9))*10

        self.prev_observation = ()
        self.prev_action = 0 
        self.prev_reward = 0

        self.step = 0
        self.game = 0

    def reset(self):
        """Reset the internal state of the agent, for a new run.

        You need to reset the internal state of your agent here, as it
        will be started in a new instance of the environment.

        You must **not** reset the learned parameters.
        """
        self.step = 0
        self.game += 1

        pass

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.
        
        Here, for the Wumpus world, observation = ((x,y), smell, breeze, charges)
        """   
        if self.game < 900:

            if np.random.uniform(low=0.0, high=1.0,)<1-self.epsilon:
                action = np.argmax(self.Q[observation[0][0], observation[0][1], 1*observation[1], 1*observation[2], observation[3],])+1
            else:
                action = np.random.randint(1,9)
        else:
            action = np.argmax(self.Q[observation[0][0], observation[0][1], 1*observation[1], 1*observation[2], observation[3],])+1


        return action    

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn. 
        """
        if self.step <1:
            self.prev_observation = observation
            self.prev_action = action
            self.prev_reward = reward

        self.max_q = np.max(self.Q[observation[0][0], observation[0][1], 1*observation[1], 1*observation[2], observation[3],])

        if (reward == 100 or reward == -10):

            self.Q[observation[0][0], observation[0][1], 1*observation[1], 1*observation[2], observation[3], action-1] = (1-self.alpha)*self.Q[observation[0][0], observation[0][1], 1*observation[1], 1*observation[2], observation[3], action-1] + self.gamma*reward 

        else : 

            self.Q[self.prev_observation[0][0], self.prev_observation[0][1], 1*self.prev_observation[1], 1*self.prev_observation[2], self.prev_observation[3], self.prev_action-1] = self.Q[self.prev_observation[0][0], self.prev_observation[0][1], 1*self.prev_observation[1], 1*self.prev_observation[2], self.prev_observation[3], self.prev_action-1] + self.alpha*(self.prev_reward + self.gamma*self.max_q - self.Q[self.prev_observation[0][0], self.prev_observation[0][1], 1*self.prev_observation[1], 1*self.prev_observation[2], self.prev_observation[3], self.prev_action-1])

        if self.step > 0:
            self.prev_observation = observation
            self.prev_action = action
            self.prev_reward = reward

        self.step +=1

        pass



class DNSAgent:
    def __init__(self):
        """Init a new agent.
        """
        self.epsilon = 0.1
        self.gamma = 0.999    
        self.alpha = 0.1

        self.Q = np.ones((100,100,2,2,5,9))*10

        self.prev_observation = ()
        self.prev_action = 0 
        self.prev_reward = 0

        self.step = 0
        self.game = 0

    def reset(self):
        """Reset the internal state of the agent, for a new run.

        You need to reset the internal state of your agent here, as it
        will be started in a new instance of the environment.

        You must **not** reset the learned parameters.
        """
        self.step = 0
        self.game += 1

        pass

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.
        
        Here, for the Wumpus world, observation = ((x,y), smell, breeze, charges)
        """ 

        if self.game < 900:

            if np.random.uniform(low=0.0, high=1.0,)<1-self.epsilon:

                if observation[1]:
                    action = np.argmax(self.Q[observation[0][0], observation[0][1], 1*observation[1], 1*observation[2], observation[3],])+1
                else:
                    action = np.argmax(self.Q[observation[0][0], observation[0][1], 1*observation[1], 1*observation[2], observation[3],][0:4])+1

            else:
                if observation[1]:
                    action = np.random.randint(1,9)
                else:
                    action = np.random.randint(1,5)
        else:
            if observation[1]:
                action = np.argmax(self.Q[observation[0][0], observation[0][1], 1*observation[1], 1*observation[2], observation[3],])+1
            else:
                action = np.argmax(self.Q[observation[0][0], observation[0][1], 1*observation[1], 1*observation[2], observation[3],][0:4])+1

        return action    

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn. 
        """
        if self.step <1:
            self.prev_observation = observation
            self.prev_action = action
            self.prev_reward = reward

            self.step +=1

            pass

        self.max_q = np.max(self.Q[observation[0][0], observation[0][1], 1*observation[1], 1*observation[2], observation[3],])

        if (reward == 100 or reward == -10):

            self.Q[observation[0][0], observation[0][1], 1*observation[1], 1*observation[2], observation[3], action-1] = (1-self.alpha)*self.Q[observation[0][0], observation[0][1], 1*observation[1], 1*observation[2], observation[3], action-1] + self.gamma*reward 

        else : 

            self.Q[self.prev_observation[0][0], self.prev_observation[0][1], 1*self.prev_observation[1], 1*self.prev_observation[2], self.prev_observation[3], self.prev_action-1] = self.Q[self.prev_observation[0][0], self.prev_observation[0][1], 1*self.prev_observation[1], 1*self.prev_observation[2], self.prev_observation[3], self.prev_action-1] + self.alpha*(self.prev_reward + self.gamma*self.max_q - self.Q[self.prev_observation[0][0], self.prev_observation[0][1], 1*self.prev_observation[1], 1*self.prev_observation[2], self.prev_observation[3], self.prev_action-1])

        if self.step > 0:
            self.prev_observation = observation
            self.prev_action = action
            self.prev_reward = reward

        self.step +=1

        pass


Agent = DNSAgent