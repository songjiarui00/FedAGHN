# generate non-iid data
python generate_cifar10.py noniid - dir

# Local
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py -lbs 64 -nc 20 -jr 1 -lr 0.01 -ls 5 -nb 10 -data Cifar10 -m fedavgcnnwithbn -algo Local -gr 200 -did 0 --seed_fixed 9999 > output_cifar10_dir01_9999Local.out 2>&1 &
# FedAvg and FedAvg-FT
CUDA_VISIBLE_DEVICES=1 nohup python -u main.py -lbs 64 -nc 20 -jr 1 -lr 0.01 -ls 5 -nb 10 -data Cifar10 -m fedavgcnnwithbn -algo FedAvg -gr 200 -did 0 --seed_fixed 9999 > output_cifar10_dir01_9999FedAvg.out 2>&1 &
# FedAGHN
CUDA_VISIBLE_DEVICES=2 nohup python -u main.py -lbs 64 -nc 20 -jr 1 -lr 0.01 -ls 5 -nb 10 -data Cifar10 -m fedavgcnnwithbn -algo FedAGHN -gr 200 -did 0 --hn_lr 0.005 --beta_hn 0.5 --gama_hn 0.03 --gama_position outer --gama_threshold 0.0 --gama_outer_policy outer_clamp --gama_modify unfixed --beta_modify unfixed --seed_fixed 9999 > output_cifar10_dir01_9999FedAGHN_05_003.out 2>&1 &

