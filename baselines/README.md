# Convolutional flow (comparison in Appendix D of the paper)

The convolutional flow of the paper has 16 affine coupling layers with checkerboard masks and convolutional networks
of two hidden layers with 8 channels and kernel size 3 (the configuration of the tutorial of Albergo et al.,
Phys. Rev. D 103, 074504 (2021)). Its density is the Z2 mixture q_mix(phi) = [q(phi) + q(-phi)] / 2, and it is
trained with the reverse KL divergence.

```bash
# train (defaults: 2500 steps, batch 64, Adam with learning rate 1e-3)
python baselines/nf/train_nf_phi4_mixture.py --kappa 0.2705 --L 16 --seed 0 --device cpu \
    --out baselines/runs/k0.2705/nf_L16_mix_s0.pt
# proposals, reweighting, IMH chains and ESS
python baselines/nf/eval_rkl_nf.py --ckpt $PWD/baselines/runs/k0.2705/nf_L16_mix_s0 --mode mix
```

`--epochs 20000` gives the eight times longer training and `--hidden_sizes 32 32` the wider network of Appendix D.
