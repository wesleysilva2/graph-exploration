import gymnasium as gym
import pickle
import networkx as nx
import numpy as np
import random

class GraphExplorationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, network: nx.Graph, actions_amout: int, stopClass = None, rewardClass = None, initial = None, target = None):
        self.network = network 

        self.initial = list(self.network.nodes)[random.randint(0, self.network.number_of_nodes()-1)] if initial is None else initial
        self.state = self.initial
        self.target = list(self.network.nodes)[random.randint(0, self.network.number_of_nodes()-1)] if target is None else target

        while(self.state == self.target and self.network.number_of_nodes != 1):
            self.target = list(self.network.nodes)[random.randint(0, self.network.number_of_nodes()-1)]


        self.stop = DefaultStopClass() if stopClass is None else stopClass 
        self.reward = DefaultReward() if rewardClass is None else rewardClass 

        self.action_space = gym.spaces.Discrete(actions_amout)
        self.observation_space = gym.spaces.Box(low=0, high=np.array([self.network.number_of_nodes() - 1 , self.network.number_of_nodes() - 1]), shape=(2,), dtype=np.int64)

        self.count = 0
        
    def reset(self,seed=None):
        super().reset(seed=seed)
        self.initial = list(self.network.nodes)[random.randint(0, self.network.number_of_nodes()-1)]
        self.state = self.initial
        list(self.network.nodes)[random.randint(0, self.network.number_of_nodes()-1)]


        while(self.state == self.target and self.network.number_of_nodes != 1):
            self.target = list(self.network.nodes)[random.randint(0, self.network.number_of_nodes()-1)]
        
        self.count = 0

        obs = (self.state, self.target)
        return obs, {}
    
    

    def step(self,action):
        self.count += 1

        possibleNextStates = list(self.network.neighbors(self.state))
        previousState = self.state

        if len(possibleNextStates) != 0:
            self.state = possibleNextStates[action]

        reward = self.reward.getReward(self.state, previousState, action, self.target, self.network)

        terminated = self.stop.isTerminated(self.state, previousState, action, self.target, self.network)

        obs = (self.state, self.target)

        return obs, reward, terminated, False, {"count" : self.count}


class RewardBaseClass():
    def getReward(self, state, previousState, action, target, graph):
        raise NotImplementedError

class StopConditionBaseClass():
    def isTerminated(self, state, previousState, action, target, graph):
        raise NotImplementedError

class DefaultReward(RewardBaseClass):
    def __init__(self) -> None:
        super().__init__()
        with open('./output/combined_sum_amount.pkl', 'rb') as f:
            self.waitTimeDict = pickle.load(f) 


    def getReward(self, state, previousState, action, target, graph):

        totalTime = self.waitTimeDict[(previousState, state)][0]
        amount = self.waitTimeDict[(previousState, state)][1]

        reward = -totalTime / amount

        if state == target:
            reward += 10000000

        return reward
    
class DefaultStopClass(StopConditionBaseClass):
    def isTerminated(self, state, previousState, action, target, graph):
        return state == target
    

if __name__ == "__main__":
    
    def run_q(episodes, env, alpha, gamma, epsilon, is_training=True):
        import time
        import matplotlib.pyplot as plt
        startTime = time.time()
        try:
            if(is_training):
                q = {}
            else:
                f = open('q.pkl', 'rb')
                q = pickle.load(f)
                f.close()
                
            stepsPerEpisode = np.zeros(episodes)
            stepCount = 0
        
            for i in range(episodes):
                state = env.reset()[0]
                start = state[0]
                target = state[1]
                terminated = False

                while(not terminated):

                    if state not in q:
                        q[state] = {}
                        for act in range(len(list(env.network.neighbors(state[0])))):
                            q[state][act] = 0    

                    a = random.random()    

                    if is_training and a< epsilon:
                        n = list(env.network.neighbors(state[0]))
                        action = random.randint(0, max(0,len(n) -1)) # based on number of neighbors of position node
                    else:             
                        values = [ q[state][act] for act in range(len(list(env.network.neighbors(state[0])))) ]
                        ix_max = [ act for act, value in enumerate(values) if value == max(values) ]
                        action = ( random.choice(ix_max) ) if len(ix_max) > 0 else 0
                    
                    new_state,reward,terminated,_,extra = env.step(action)
                    
                    if new_state not in q:
                        q[new_state] = {}
                        for act in range(len(list(env.network.neighbors(new_state[0])))):
                            q[new_state][act] = 0

                    if is_training:
                        qValues = [ q[new_state][act] for act in range(len(list(env.network.neighbors(new_state[0])))) ]

                        sample = reward + gamma * (max(qValues) if len(qValues) > 0 else 0)

                        part1 = ( (1 - alpha) * q[state][action] ) if len(q[state]) > 0 else float('-inf')
                        part2 = alpha * sample
                        
                        q[state][action] = part1 + part2

                    state = new_state

                    stepCount += 1

                    if terminated:
                        print(f"({i}) {start} -> {target} in {stepCount} steps with epsilon {epsilon}")
                        stepsPerEpisode[i] = stepCount
                        stepCount = 0
                
                epsilon = max(epsilon - 1/(episodes), 0.01)
                if time.time() - startTime > 60 * 60 * 2: # 2 hours
                    break

            env.close()

        finally:
            sumSteps = np.zeros(i)
            for t in range(i):
                sumSteps[t] = np.mean(stepsPerEpisode[max(0, t-100):(t+1)]) 

            plt.plot(sumSteps)
            plt.savefig(f'q{env.initial}-{env.target}-{alpha}-{gamma}-{epsilon}.png')

            if is_training:
                # Save Q Table
                f = open(f'q{env.initial}-{env.target}-{alpha}-{gamma}-{epsilon}.pkl',"wb")
                pickle.dump(q, f)
                f.close()


    with open('./sunt/graph_designer/graph_gtfs.gpickle', 'rb') as f:
        G = pickle.load(f)

    
    env = GraphExplorationEnv(G, 9)

    run_q(1000, env, 0.9, 0.9, 0.1, is_training=True)
