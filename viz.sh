# Description: Script to visualize attention maps of DINO model
# Usage: bash dino/viz.sh
# Author: Luke Byrne
# Created: 2024-02-13

python3 visualize_attention.py --pretrained_weights ./model_weights/dinov2_deitsmall16_pretrain.pth \
                                --arch vit_small \
                                --patch_size 16 \
                                --output_dir ./output \
                                --image_path ../Littleton_Sample_Data/1029.png \
                                --image_size 224 \

