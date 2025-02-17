export PYTHONPATH="$PYTHONPATH:$PWD/emma"
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --model.vision_backbone_id "dinosiglip-vit-so-384px" \
  --model.image_resize_strategy "resize-naive" \
  --model.llm_backbone_id "mamba-2.8b-zephyr" \
  --model.type "EMMA+3b" \
  --model.finetune_global_batch_size 128 \
  --model.finetune_per_device_batch_size 1 \
  --dataset.type "llava-lvis4v-lrv" \
  --run_id "EMMA-V1"

