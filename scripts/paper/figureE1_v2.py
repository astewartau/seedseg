#!/usr/bin/env python3
"""Supplementary Figure E1: nnU-Net training curves of the five cross-validation folds (511).
usage: figureE1_v2.py --out <dir>
"""
import argparse, glob, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

R = '/scratch/user/uqaste15/prostate-v2/nnunet/nnUNet_results/Dataset511_SeedSegCvAll_corrected/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres'


def parse(f):
    tr, va, dice = [], [], []
    for line in open(f):
        if ': train_loss ' in line: tr.append(float(line.split()[-1]))
        elif ': val_loss ' in line: va.append(float(line.split()[-1]))
        elif ': Pseudo dice ' in line: dice.append([float(x) for x in re.findall(r'\(([-0-9.e]+)\)', line)])
    return np.array(tr), np.array(va), np.array(dice)


ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
for f in range(5):
    tr, va, d = parse(glob.glob(f'{R}/fold_{f}/training_log_*.txt')[0])
    ax1.plot(tr, color='#1f5f8b', lw=0.6, alpha=0.8, label='Training' if f == 0 else None)
    ax1.plot(va, color='#e07b00', lw=0.6, alpha=0.8, ls='--', label='Validation' if f == 0 else None)
    k = np.ones(25) / 25
    ax2.plot(np.convolve(d[:, 0], k, 'valid'), color='#1f5f8b', lw=0.8, label='Marker' if f == 0 else None)
    ax2.plot(np.convolve(d[:, 1], k, 'valid'), color='#6a9e3f', lw=0.8, ls='--', label='Prostate' if f == 0 else None)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss (Dice + cross-entropy)'); ax1.legend(frameon=False, fontsize=8)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Validation Dice (25-epoch mean)'); ax2.set_ylim(0, 1); ax2.legend(frameon=False, fontsize=8)
for ax, l in zip((ax1, ax2), 'AB'): ax.text(-0.15, 1.02, l, transform=ax.transAxes, fontsize=11, fontweight='bold')
fig.tight_layout(); fig.savefig(f'{a.out}/FigureE1.png', dpi=300); fig.savefig(f'{a.out}/FigureE1.tiff', dpi=300)
