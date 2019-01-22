CUDA_VISIBLE_DEVICES=0 python loaddata.py -d market -a resnet50 -b 64 -j 4 --epochs 60 --log logs/market-1501/PCB/ --combine-trainval --feature 256 --height 256 --width 256 --step-size 40 --data-dir ~/HumpbackWhale/dataset/
 
