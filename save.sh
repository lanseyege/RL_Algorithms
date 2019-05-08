envs=('Hopper-v2' 'Walker2d-v2' 'Reacher-v2' 'InvertedPendulum-v2' 'InvertedDoublePendulum-v2' 'Ant-v2' 'Humanoid-v2' 'HalfCheetah-v2' )
t=7
env=${envs[$t]}
num=200
mm=5
echo $env

nohup time python train.py --als save --vv 5 --max_expert_num 50000 --env_id $env --seed 1 --use_cuda True > res/"$env"_save & 
