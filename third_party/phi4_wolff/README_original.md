# phi4-wolff

二维 \(\phi^4\) 的 **Wolff + HMC** 采样代码。默认作用量 \(\lambda=0.022\)。

`algorithm=wolff_hmc` 每 `wolff_every` 条 HMC 轨迹插入一次 Wolff 团簇更新；`algorithm=hmc` 是对照。

## 文件

- `src/action.py` — \(\phi^4\) 作用量与力
- `src/hmc_wolff_tau_phi4_fullsave.py` — 固定步长 HMC / Wolff+HMC 驱动
- `src/run_hmc_wolff_tau_fixed_all.sh` — 大格子队列（\(8\times16\) 到 \(8\times4096\)）
- `src/build_phi4_wolff_reference_audit.py` — 把固定 \(\kappa\) 估计接到 Wolff-TI 参考上

依赖：`numpy`、`torch`、`numba`、`scipy`。`pyerrors` 可选。

## 运行

```bash
python src/hmc_wolff_tau_phi4_fullsave.py \
  --algorithm wolff_hmc --Lx 8 --Lt 256 --kappa 0.27 \
  --out-root ./out --n-chains 1 --n-steps 5000 --burn-in 500 \
  --wolff-every 5 --device cuda
```

大格子原始场配置不在这个仓库里。
