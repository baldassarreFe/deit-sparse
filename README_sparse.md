# DeiT with sparse activations

Create environment:
```bash
mamba create -y -n deit -c pytorch -c conda-forge \
  python tqdm jupyterlab jupyter_console ipywidgets \
  pytorch=1.7.0 torchvision=0.8.1 cudatoolkit-dev=11.0 cudnn \
  torchmetrics einops opt_einsum \
  numpy pandas matplotlib seaborn tabulate \
  scikit-learn scikit-image pillow

conda activate deit
conda env config vars set BETTER_EXCEPTIONS=1
pip install timm==0.3.2 submitit better_exceptions tensorboard entmax

git clone https://github.com/NVIDIA/apex ~/apex
cd ~/apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Train models with sparse activations:
```bash
COMMON_ARGS=(
  --no-model-ema
  --data-set IMNET
  --data-path /datasets01/imagenet_full_size/061417
  --find_unused_parameters
  --epochs 800
  --weight-decay 0.05
  --sched cosine
  --input-size 224
  --reprob 0.0
  --smoothing 0.0
  --warmup-epochs 5
  --drop 0.0
  --seed 0
  --opt adamw
  --warmup-lr 1e-6
  --mixup .8
  --cutmix 1.0
  --unscale-lr
  --repeated-au
  --bce-loss
  --color-jitter 0.3
  --ThreeAugment
)

DIST_ARGS=(
  --partition learnfair
  --nodes 1
  --ngpus 8
  --use_volta32
)

# SLURM
for ACTIVATION in sparsemax entmax15 alphaentmax; do
  python run_with_submitit.py ${COMMON_ARGS[@]} ${DIST_ARGS[@]} --model "deit_tiny_patch16_LS_${ACTIVATION}"   --lr 4e-3 --drop-path 0.05 --batch 256
  python run_with_submitit.py ${COMMON_ARGS[@]} ${DIST_ARGS[@]} --model "deit_small_patch16_LS_${ACTIVATION}"  --lr 4e-3 --drop-path 0.05 --batch 128
  python run_with_submitit.py ${COMMON_ARGS[@]} ${DIST_ARGS[@]} --model "deit_medium_patch16_LS_${ACTIVATION}" --lr 3e-3 --drop-path 0.20 --batch 128
  python run_with_submitit.py ${COMMON_ARGS[@]} ${DIST_ARGS[@]} --model "deit_base_patch16_LS_${ACTIVATION}"   --lr 3e-3 --drop-path 0.20 --batch 64
done

# Local debug
python main.py ${COMMON_ARGS[@]} --model "deit_tiny_patch16_LS_alphaentmax" --lr 4e-3 --drop-path 0.05
```

Download and extract ImageNet and [ImageNet-C](https://github.com/hendrycks/robustness):
```bash
mkdir -p data
cd data
for DATASET in blur digital extra noise weather; do
  wget "https://zenodo.org/record/2235448/files/${DATASET}.tar?download=1" -O "${DATASET}.tar"
  mkdir "${DATASET}"
  tar xf "${DATASET}.tar" -C "${DATASET}"
done
```

Organize `data` as such e.g. with symlinks:
```
data
├── imagenet
│   └── val/1
├── noise
│   ├── gaussian_noise/{1,2,3,4,5}
│   ├── impulse_noise/{1,2,3,4,5}
│   └── shot_noise/{1,2,3,4,5}
├── blur
│   ├── defocus_blur/{1,2,3,4,5}
│   ├── glass_blur/{1,2,3,4,5}
│   ├── motion_blur/{1,2,3,4,5}
│   └── zoom_blur/{1,2,3,4,5}
├── weather
│   ├── brightness/{1,2,3,4,5}
│   ├── fog/{1,2,3,4,5}
│   ├── frost/{1,2,3,4,5}
│   └── snow/{1,2,3,4,5}
├── extra
│   ├── gaussian_blur/{1,2,3,4,5}
│   ├── saturate/{1,2,3,4,5}
│   ├── spatter/{1,2,3,4,5}
│   └── speckle_noise/{1,2,3,4,5}
└── digital
    ├── contrast/{1,2,3,4,5}
    ├── elastic_transform/{1,2,3,4,5}
    ├── jpeg_compression/{1,2,3,4,5}
    └── pixelate/{1,2,3,4,5}
```

Download softmax checkpoints:
```bash
mkdir -p checkpoints
cd checkpoints
wget 'https://dl.fbaipublicfiles.com/deit/deit_3_small_224_1k.pth'  -O 'deit_small_patch16_LS.pth'
wget 'https://dl.fbaipublicfiles.com/deit/deit_3_medium_224_1k.pth' -O 'deit_medium_patch16_LS.pth'
wget 'https://dl.fbaipublicfiles.com/deit/deit_3_base_224_1k.pth'   -O 'deit_base_patch16_LS.pth'
```

Organize `checkpoints` as such, e.g. with symlinks:
```
checkpoints
├── deit_tiny_patch16_LS_sparsemax.pth
├── deit_tiny_patch16_LS_entmax15.pth
├── deit_tiny_patch16_LS_alphaentmax.pth
├── deit_small_patch16_LS.pth
├── deit_small_patch16_LS_sparsemax.pth
├── deit_small_patch16_LS_entmax15.pth
├── deit_small_patch16_LS_alphaentmax.pth
├── deit_medium_patch16_LS.pth
├── deit_medium_patch16_LS_sparsemax.pth
├── deit_medium_patch16_LS_entmax15.pth
├── deit_medium_patch16_LS_alphaentmax.pth
├── deit_base_patch16_LS.pth
├── deit_base_patch16_LS_sparsemax.pth
├── deit_base_patch16_LS_entmax15.pth
└── deit_base_patch16_LS_alphaentmax.pth
```

Evaluate:
```bash
COMMON_ARGS=(
  --eval
  --data-set FOLDER
  --no-model-ema
)

DATASETS=(
  imagenet/val/1
  noise/{gaussian_noise,shot_noise,impulse_noise}/{1..5}
  blur/{defocus_blur,glass_blur,motion_blur,zoom_blur}/{1..5}
  weather/{frost,snow,fog,brightness}/{1..5}
  digital/{contrast,elastic_transform,pixelate,jpeg_compression}/{1..5}
  extra/{speckle_noise,spatter,gaussian_blur,saturate}/{1..5}
)

for SIZE in tiny small medium base; do
  case "${SIZE}" in
    tiny | small)
      BATCH_SIZE=2048;;
    medium)
      BATCH_SIZE=1024;;
    base)
      BATCH_SIZE=512;;
  esac

  for ACTIVATION in '' _sparsemax _entmax15 _alphaentmax; do
    MODEL="deit_${SIZE}_patch16_LS${ACTIVATION}"
    if [[ ! -f "checkpoints/${MODEL}.pth" ]]; then
      continue
    fi

    for DATASET in ${DATASETS[*]}; do
      if [[ ! -d "data/${DATASET}" ]]; then
        continue
      fi

      python main.py ${COMMON_ARGS[@]} \
        --data-path "data/${DATASET}" --batch "${BATCH_SIZE}" \
        --model "${MODEL}" --resume "checkpoints/${MODEL}.pth" \
        --output_dir "results/${MODEL}"

    done
  done
done
```
