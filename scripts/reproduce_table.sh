CUDA_VISIBLE_DEVICES=0 python src/eval.py --model_path model/alanine/pot_fix.pt --bias_scale 20000 --num_steps 500

# CUDA_VISIBLE_DEVICES=1 python src/eval.py --model_path model/alanine/force_fix.pt --force --bias_scale 1000 --num_steps 500

# CUDA_VISIBLE_DEVICES=2 python src/eval.py --model_path model/histidine/pot_flex.pt --sigma 0.1 --molecule histidine

# CUDA_VISIBLE_DEVICES=3 python src/eval.py --model_path model/histidine/force_flex.pt --force --bias_scale 0.0001 --sigma 0.1 --molecule histidine
