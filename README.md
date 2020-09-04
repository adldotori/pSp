### Reference
<https://github.com/rosinality/stylegan2-pytorch>
<https://github.com/S-aiueo32/lpips-pytorch>
<https://github.com/TreB1eN/InsightFace_Pytorch>

python -m torch.distributed.launch --nproc_per_node=4 --master_port=8800 train.py --batch 4 --ckpt config/stylegan2-kceleb-config-f.pt