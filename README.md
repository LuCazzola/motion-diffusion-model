# Fork of: MDM - Human Motion Diffusion Model

```
python3 -m sample.generate \
  --model_path ./save/humanml_enc_512_50steps/model000750000.pt \
  --few_shot \
  --action_labels 0 1 2 3 4 \
  --dataset_desc ../../data/NTU_RGBD/ntu_desc.json
```