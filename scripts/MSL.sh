export CUDA_VISIBLE_DEVICES=0

python main_prediction.py --num_epochs 10    --batch_size 256 --d_model 16  --n_heads 16  --lr 1e-5 --mode train --dataset MSL  --data_path MSL --input_c 1    --output_c 1  --loss_fuc MSE  --win_size 64 --patch_size 248
python main_prediction.py --num_epochs 10    --batch_size 256 --d_model 16   --n_heads 16  --lr 1e-5 --mode test    --dataset MSL   --data_path MSL  --input_c 1    --output_c 1  --loss_fuc MSE  --win_size 64  --patch_size 248
