save_dir="./results/Anytime-GE-ImageNet-step=4-block=5-growthRate=16"

python main_GE.py \
--data-root [PATH_TO_IMAGENET] \
--save $save_dir \
--data ImageNet \
--gpu 0,1,2,3 \
--arch msdnet_ge \
--batch-size 256 \
--epochs 90 \
--nBlocks 5 \
--stepmode even \
--growthRate 16 \
--grFactor 1-2-4-4 \
--bnFactor 1-2-4-4 \
--step 4 \
--base 4 \
--nChannels 32 \
--use-valid \
-j 8 