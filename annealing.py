import numpy as np

#Annealed Stein Variational Gradient Descent (D’Angelo F. and Fortuin V.) (2021)
#https://arxiv.org/pdf/2101.09815.pdf

class anneal_sets():
    def __init__(self, num_episode, update_frequency, C=10, p=1, 
                                        anneal_method="cyclical"):

        #Hyper Parameters
        self.T = num_episode/update_frequency
        self.p = p
        self.linear_anneal_factor = 1/(0.8*self.T)
        self.C = C
        
        #Anneal Method
        if anneal_method == "cyclical":
            self.annealing = self.cyclical_annealing
        elif anneal_method == "tanh":
            self.annealing = self.tanh_annealing
        elif anneal_method == "linear":
            self.annealing = self.linear_annealing
            
    
    def linear_annealing(self, t):
        tmp = 0.1+t*self.linear_anneal_factor
        # t is number of current update
        if tmp <= 0.5:
            return tmp
        else:
            return 0.5

    def tanh_annealing(self, t):
        # t is number of current update
        return np.tanh((1.3*t/self.T)**self.p)
    
    def cyclical_annealing(self, t):
        # t is number of current update
        ratio = self.T/self.C
        return (np.mod(t,ratio)/ratio)**self.p