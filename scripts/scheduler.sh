python train_net.py --dataset nvidia --data-path /media/ws1/Data3/datasets/ --initial --max-iter 2200 --num-gpus 4 --lr 0.000025
python train_net.py --dataset dopose --data-path /media/ws1/Data3/datasets/ --max-iter 800 --num-gpus 4 --lr 0.000025
# python train_net.py --dataset sps --data-path ~/BlenderProc/sps_synthetic --max-iter 500 --num-gpus 4
python segment_dataset.py --model-path ../output/model_final.pth