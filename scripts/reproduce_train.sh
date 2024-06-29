CUDA_VISIBLE_DEVICES=0 python src/baseline_setup_train.py --seed 0 --save_dir results/alanine/pot/0/ --bias_scale 20 
# CUDA_VISIBLE_DEVICES=1 python src/baseline_setup_train.py --seed 1 --save_dir results/alanine/pot/1/ --bias_scale 20 &
# CUDA_VISIBLE_DEVICES=2 python src/baseline_setup_train.py --seed 2 --save_dir results/alanine/pot/2/ --bias_scale 20 &
# CUDA_VISIBLE_DEVICES=3 python src/baseline_setup_train.py --seed 3 --save_dir results/alanine/pot/3/ --bias_scale 20 &
# CUDA_VISIBLE_DEVICES=4 python src/baseline_setup_train.py --seed 0 --save_dir results/alanine/force/0/ --force &
# CUDA_VISIBLE_DEVICES=5 python src/baseline_setup_train.py --seed 1 --save_dir results/alanine/force/1/ --force &
# CUDA_VISIBLE_DEVICES=6 python src/baseline_setup_train.py --seed 2 --save_dir results/alanine/force/2/ --force &
# CUDA_VISIBLE_DEVICES=7 python src/baseline_setup_train.py --seed 3 --save_dir results/alanine/force/3/ --force 
