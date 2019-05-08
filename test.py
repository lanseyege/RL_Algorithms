
import argparse
parser = argparse.ArgumentParser(description='run RL algorithms...')
parser.add_argument('--env_id', default='a', help='enviroment name of Mujoco')
parser.add_argument('--seed', default=1, type=int, help='random seed ')

args = parser.parse_args()

print(args)

from lib.frechetdist import frdist

P=[[1,1], [2,1], [2,2]]
Q=[[2,2], [0,1], [2,4]]
print(frdist(P,Q))

P=[[1,1], [2,1], [2,2]]
Q=[[1,1], [2,1], [2,2]]

print(frdist(P,Q))
