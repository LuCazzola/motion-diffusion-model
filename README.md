# Fork of: MDM - Human Motion Diffusion Model

* Customized implementation of ![MDM - Motion Diffusion Model](https://github.com/GuyTevet/motion-diffusion-model)
* Reference ![Base Repo](https://github.com/LuCazzola/Few-Shot_MDM) to have a better understanding of the context

# Key modifications

1. Added $n$-ways $m$-shots Text-2-Motion (t2m) generation:
  * A .json is provided listing a set of action classes we'd like to synthetize. To each action class a set of natural language descriptions is associated.
  * On each shot (repetition) $n$ samples are synthetized by random sampling a caption from the candidate captions set
  * In ![Base Repo](https://github.com/LuCazzola/Few-Shot_MDM) we use such synthetic samples to train a HAR model

```
python -m sample.generate \
  --model_path ./save/humanml_enc_512_50steps/model000750000.pt \
  --few_shot \
  --action_labels 0 1 2 3 4 \
  --dataset_desc ../../data/NTU_RGBD/ntu_desc.json
```

2. Added Few-Shot training on NTU60

```
python -m train.train_mdm \
  --dataset ntu60 \
  --split splits/fewshot/5way_10shot_seed19/xset/train \
  --save_dir save/my_few_shot_ntu60_trans_enc_512 \
  --diffusion_steps 50 \
  --few_shot \
  --mask_frames \
  --use_ema
```