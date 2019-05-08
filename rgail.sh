#python train.py --als ppo --vv 3 --max_genert_num 200 --env_id HalfCheetah-v2
#python train.py --als trpo --vv 3 --max_genert_num 200 #--env_id HalfCheetah-v2
#python train.py --als save --vv 3 --max_expert_num 50000 --env_id HalfCheetah-v2
#default action=0  seed=1
envs=('Hopper-v2' 'Walker2d-v2' 'Reacher-v2' 'InvertedPendulum-v2' 'InvertedDoublePendulum-v2' 'Ant-v2' 'Humanoid-v2' 'HalfCheetah-v2')
t=-2
env=${envs[$t]}
num=10000
vv=5
mm=10
echo $env
#nohup time python train.py --als gail --vv $vv --mm $mm --max_genert_num $num --env_id $env --action 0 --seed 1 --use_cuda True > res/"$env"_gail1 & 
#nohup time python train.py --als gail --vv $vv --mm $mm --max_genert_num $num --env_id $env --action 0 --seed 2 --use_cuda True > res/"$env"_gail2 & 
nohup time python train.py --als gail --vv $vv --mm $mm --max_genert_num $num --env_id $env --action 0 --seed 3 --use_cuda True > res/"$env"_gail3 & 
nohup time python train.py --als gail --vv $vv --mm $mm --max_genert_num $num --env_id $env --action 0 --seed 4 --use_cuda True > res/"$env"_gail4 & 
#nohup time python train.py --als gail --vv $vv --mm $mm --max_genert_num $num --env_id $env --action 1 --seed 1 --use_cuda True > res/"$env"_gail_noact1 & 
#nohup time python train.py --als gail --vv $vv --mm $mm --max_genert_num $num --env_id $env --action 1 --seed 2 --use_cuda True > res/"$env"_gail_noact2 & 
#nohup time python train.py --als gail --vv $vv --mm $mm --max_genert_num $num --env_id $env --action 1 --seed 3 --use_cuda True > res/"$env"_gail_noact3 & 
#nohup time python train.py --als gail --vv $vv --mm $mm --max_genert_num $num --env_id $env --action 1 --seed 4 --use_cuda True > res/"$env"_gail_noact4 & 
#nohup time python train.py --als gail --vv $vv --mm $mm --max_genert_num $num --env_id $env --action 2 --seed 1 --use_cuda True > res/"$env"_gail_agact1 & 
#nohup time python train.py --als gail --vv $vv --mm $mm --max_genert_num $num --env_id $env --action 2 --seed 2 --use_cuda True > res/"$env"_gail_agact2 & 
#nohup time python train.py --als gail --vv $vv --mm $mm --max_genert_num $num --env_id $env --action 2 --seed 3 --use_cuda True > res/"$env"_gail_agact3 & 
#nohup time python train.py --als gail --vv $vv --mm $mm --max_genert_num $num --env_id $env --action 2 --seed 4 --use_cuda True > res/"$env"_gail_agact4 & 
#nohup time python train.py --als gail --vv $vv --mm $mm --max_genert_num $num --env_id $env --action 3 --seed 1 --use_cuda True > res/"$env"_gail_states1 & 
#nohup time python train.py --als gail --vv $vv --mm $mm --max_genert_num $num --env_id $env --action 3 --seed 2 --use_cuda True > res/"$env"_gail_states2 & 
#nohup time python train.py --als gail --vv $vv --mm $mm --max_genert_num $num --env_id $env --action 3 --seed 3 --use_cuda True > res/"$env"_gail_states3 & 
#nohup time python train.py --als gail --vv $vv --mm $mm --max_genert_num $num --env_id $env --action 3 --seed 4 --use_cuda True > res/"$env"_gail_states4 & 
