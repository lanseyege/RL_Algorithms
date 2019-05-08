envs=('Hopper-v2' 'Walker2d-v2' 'Reacher-v2' 'InvertedPendulum-v2' 'InvertedDoublePendulum-v2' 'Ant-v2' 'Humanoid-v2' 'HalfCheetah-v2')
t=7
env=${envs[$t]}
num=200
vv=8
mm=6
echo $env
nohup time python train.py --als ppo --vv $vv --mm $mm --max_genert_num $num --env_id $env --action 0 --seed 1 --use_cuda True > res/"$env"_ppo$vv & 


#nohup time python train.py --als ppo --vv 0 --max_genert_num 200 --env_id Humanoid-v2 > res/Humanoid_ppo & 
# nohup time python train.py --als trpo --vv 3 --max_genert_num 200 #--env_id HalfCheetah-v2 > res/HalfCheetah_trpo & 
# nohup time python train.py --als save --vv 0 --max_expert_num 50000 --env_id HalfCheetah-v2 > res/HalfCheetah_save & 
