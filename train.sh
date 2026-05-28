
python main.py \
    -t \
    --base configs/train.yaml \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --logdir "logs" \
    --finetune_from sd-image-conditioned-v2.ckpt \
