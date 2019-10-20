save_dir="./results/IMTA_MSDNet-ImageNet-step=4-T=1-gamma=0.1"
pretrained_dir="./results/Anytime-GE-ImageNet-step=4-block=5-growthRate=16"
mkdir ../results/$save_dir
echo "***** copy index *****"
cp $pretrained_dir/index.pth $save_dir


python main_IMTA.py \
--data-root [PATH_TO_IMAGENET] \
--save $save_dir \
--data ImageNet \
--gpu 0,1,2,3 \
--arch IMTA_MSDNet \
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
--pretrained $pretrained_dir/save_models/model_best.pth.tar \
-T 1.0 \
--gamma 0.1 \
--use-valid \
-j 8 