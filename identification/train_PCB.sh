CUDA_VISIBLE_DEVICES=0 python2 PCB.py -d market -a resnet50 -b 64 -j 4 --epochs 60 --log logs/market-1501/PCB/ --combine-trainval --feature 256 --height 384 --width 128 --step-size 40 --data-dir ~/datasets/Market-1501/
 
