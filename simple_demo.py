from sys import path
from roll_out import rollout
import torch
#from plot import plot_learning_curve
import argparse
import os
current_dir = os.path.split(os.path.abspath(__file__))[0]  
config_path = current_dir.rsplit('/', 2)[0]  



parser = argparse.ArgumentParser(description='Kernel Comparsion Experiments')

parser.add_argument('--env_name', default='Hopper-v2', help="Environment Name of OpenAI gym",
            type = str)
parser.add_argument('--algorithm', default='SVPG', help="Choose RL Algorithms",
            type=str)
parser.add_argument('--num_agents', default=3, help="Number of Agents working in parallel", 
            type=int)
parser.add_argument('--num_iterations', default=1000, help='Number of Iterations',
            type=int)
parser.add_argument('--seed',default=17,help='ramdom seed',
            type=int)
parser.add_argument('--sm', default=3, help="smooth factor for plotting",
            type=int)
parser.add_argument("--anneal_method", default="linear", help="tempature annealing method",
            type=str)
parser.add_argument("--kernel", default="Gaussian", help="kernel for SVPG",
            type=str)
parser.add_argument('--log_save_path', default=current_dir,
            help="log_save path", type=str)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config = parser.parse_args()

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    #Run
    std_score, avg_score = rollout(config.env_name, algorithm=config.algorithm, anneal_method=config.anneal_method,
                            kernel=config.kernel, num_iteration=config.num_iterations, num_agents=config.num_agents,log_dir=config.log_save_path)
    
    #Plot
    data = {"mean":avg_score, "std":std_score, "label":"{}-{}".format(config.algorithm, config.kernel)}
    data = [data]
    #plot_learning_curve(data, env_name=config.env_name,path=current_dir, sm=config.sm)








