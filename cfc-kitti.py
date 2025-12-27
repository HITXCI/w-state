import os
import sys
import argparse
import importlib.util
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -------------------------
# CONFIG: edit these paths
# -------------------------
DATA_CSV_PATHS = [
    '/home/rrc/xy_projects/LLN/KITTI/catkin_lego/outputs/dataset_10.csv',
    # '/absolute/path/to/your/seq2.csv'
]
UPLOADED_CFC_PATH = "/home/rrc/xy_projects/LLN/CfC/torch_cfc.py"  

# Training defaults
BATCH_SIZE = 16
EPOCHS = 25
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 1e-3

AUG_CONFIG = {
    'gps_sigma': 1.0,             # Gaussian noise (meters)
    'gps_spike_prob_per_s': 0.005,
    'gps_spike_amp': (3.0, 12.0), # GPS jump amplitude (meters)
    # optional slow drift (random walk) on GPS to simulate bias/offset over time:
    'gps_bias_rw_sigma': 0.01     # meters * sqrt(dt)
}

COLUMN_MAP = {
    'time': 'time',
    'gps_x': 'gps_x',
    'gps_y': 'gps_y',
    'gt_x': 'gt_x',
    'gt_y': 'gt_y'
}

class FullSequenceDataset(Dataset):
    def __init__(self, csv_paths, augment=False, cfg=AUG_CONFIG):
        self.sequences = []
        for p in csv_paths:
            df = pd.read_csv(p)
            times = df['time'].values.astype(np.float32)
            gps = np.stack([df['gps_x'].values, df['gps_y'].values * 10], axis=1).astype(np.float32)
            gt = np.stack([df['gt_x'].values, df['gt_y'].values * 10], axis=1).astype(np.float32)
            gt_mask = (~np.isnan(gt[:,0]) & ~np.isnan(gt[:,1])).astype(np.float32)
            self.sequences.append({'times': times, 'gps': gps, 'gt': gt, 'gt_mask': gt_mask})
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        s = self.sequences[idx]
        gps_noisy = add_gps_noise_and_spikes(s['times'], s['gps'], AUG_CONFIG) if True else np.nan_to_num(s['gps'], nan=0.0)
        x = gps_noisy.astype(np.float32)
        gps_mask = (~np.isnan(s['gps'][:,0]) & ~np.isnan(s['gps'][:,1])).astype(np.float32)
        return {'times': torch.from_numpy(s['times']),
                'x': torch.from_numpy(x),
                'gps_mask': torch.from_numpy(gps_mask),
                'y_gt': torch.from_numpy(s['gt']),
                'gt_mask': torch.from_numpy(s['gt_mask'])}

def plot_full_validation_trajectory(model,
                                    val_dataset,
                                    device="cpu",
                                    best_model_path=None,
                                    out_dir="/mnt/data/val_full_track"):

    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from torch.utils.data import DataLoader

    # os.makedirs(out_dir, exist_ok=True)

    # 加载最优模型
    if best_model_path is not None:
        ck = torch.load(best_model_path, map_location=device)
        if isinstance(ck, dict) and "model_state_dict" in ck:
            model.load_state_dict(ck["model_state_dict"])
        else:
            model.load_state_dict(ck)
        print(f"[info] Loaded best model from: {best_model_path}")

    model.to(device)
    model.eval()

    # 整个 val dataset 一次一个序列
    dl = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    all_time = []
    all_pred = []
    all_gt = []
    all_meas = []
    all_gpsmask = []
    all_gtmask = []

    with torch.no_grad():
        for batch in dl:
            x = batch["x"].to(device)              # (1, T, 2)
            times = batch["times"].to(device)      # (1, T)
            pad_mask = batch["mask"].to(device)    # (1, T)
            y_gt = batch["y_gt"].to(device)        # (1, T, 2)
            gt_mask = batch["gt_mask"].to(device)  # (1, T)

            pad_mask_exp = pad_mask.unsqueeze(-1)  # (1, T, 1)

            pred = model.forward_sequence(x, times, pad_mask_exp)
            pred = pred.cpu().numpy()[0]
            times_np = times.cpu().numpy()[0]
            ygt = y_gt.cpu().numpy()[0]
            gt_mask_np = gt_mask.cpu().numpy()[0]
            pad_mask_np = pad_mask.cpu().numpy()[0]

            
            x_np = x.cpu().numpy()[0]
            meas_x = x_np[:,0]
            meas_y = x_np[:,1]
            gps_mask_np = batch.get("gps_mask", None)
            if gps_mask_np is not None:
                gps_mask_np = gps_mask_np.numpy()[0]
            else:
                gps_mask_np = pad_mask_np * (np.abs(meas_x)+np.abs(meas_y) > 1e-6)

            
            idx = pad_mask_np > 0.5

            all_time.append(times_np[idx])
            all_pred.append(pred[idx])
            all_gt.append(ygt[idx])
            all_meas.append(np.stack([meas_x[idx], meas_y[idx]], axis=1))
            all_gpsmask.append(gps_mask_np[idx])
            all_gtmask.append(gt_mask_np[idx])

   
    time_full = np.concatenate(all_time, axis=0)
    pred_full = np.concatenate(all_pred, axis=0)
    gt_full = np.concatenate(all_gt, axis=0)
    meas_full = np.concatenate(all_meas, axis=0)
    gpsmask_full = np.concatenate(all_gpsmask, axis=0)
    gtmask_full = np.concatenate(all_gtmask, axis=0)

    
    time_factor = 10
    plt.figure(figsize=(7,7))
    # GT
    idx_gt = gtmask_full > 0.5 # 0.5
    plt.plot(gt_full[idx_gt,0], gt_full[idx_gt,1] / time_factor, '-', label="GT", linewidth=2)

    # Pred
    plt.plot(pred_full[:,0], pred_full[:,1] / time_factor, '--', label="Pred", linewidth=2)

    # Measurements
    idx_meas = gpsmask_full > 0.5
    # plt.scatter(meas_full[idx_meas,0], meas_full[idx_meas,1], s=8, c='gray', alpha=0.5, label="Measurement")

    plt.legend()
    # plt.axis("equal")
    plt.title("Full validation trajectory (XY)")
    traj_path = os.path.join(out_dir, "validation_full_xy.png")
    # plt.savefig(traj_path, dpi=150, bbox_inches="tight")
    plt.ylim(-25, 10)
    plt.show()
    plt.close()
    print(f"[saved] XY trajectory: {traj_path}")

    
    fig, axes = plt.subplots(2,1,figsize=(12,6), sharex=True)

    # x(t)
    axes[0].plot(time_full, gt_full[:,0], label="gt_x")
    axes[0].plot(time_full, pred_full[:,0], '--', label="pred_x")
    # axes[0].scatter(time_full[idx_meas], meas_full[idx_meas,0], s=8, c='gray', alpha=0.5)
    axes[0].set_ylabel("x")

    # y(t)
    axes[1].plot(time_full, gt_full[:,1] / time_factor, label="gt_y")
    axes[1].plot(time_full, pred_full[:,1] / time_factor, '--', label="pred_y")
    # axes[1].scatter(time_full[idx_meas], meas_full[idx_meas,1], s=8, c='gray', alpha=0.5)
    axes[1].set_ylabel("y")
    axes[1].set_xlabel("time (s)")

    gt_x = gt_full[:,0]
    gt_y = gt_full[:,1] / time_factor
    pred_x = pred_full[:,0]
    pred_y = pred_full[:,1] / time_factor
    time = time_full
    result = []
    print("gt_x shape is: ", gt_x.shape)
    print("gt_y shape is: ", gt_y.shape)
    print("pred_x shape is: ", pred_x.shape)
    print("pred_x shape is: ", pred_y.shape)
    print("time shape is: ", time.shape)


    for i, x in enumerate(time):
        result.append([gt_x[i], gt_y[i], pred_x[i], pred_y[i], time[i]])
    result = np.array(result)
    header = "gt_x, gt_y, pred_x, pred_y, time"
    np.savetxt("/home/rrc/xy_projects/LLN/weight_kitti/result_seq_02.txt", result, delimiter=",", header=header)

    plt.legend()
    tseries_path = os.path.join(out_dir, "validation_full_timeseries.png")
    # plt.savefig(tseries_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"[saved] time-series: {tseries_path}")

    
    df = pd.DataFrame({
        "time": time_full,
        "pred_x": pred_full[:,0],
        "pred_y": pred_full[:,1],
        "gt_x": gt_full[:,0],
        "gt_y": gt_full[:,1],
        "meas_x": meas_full[:,0],
        "meas_y": meas_full[:,1],
        "gps_valid": gpsmask_full,
        "gt_valid": gtmask_full
    })
    csv_path = os.path.join(out_dir, "validation_full_track.csv")
    # df.to_csv(csv_path, index=False)
    print(f"[saved] full validation CSV: {csv_path}")

    return df

def visualize_on_val(model,
                     val_dataset,
                     device='cpu',
                     best_model_path=None,
                     out_dir='/mnt/data/val_vis',
                     n_examples=6,
                     figsize=(10,4)):

    os.makedirs(out_dir, exist_ok=True)
    # load best model if path provided
    if best_model_path is not None:
        ck = torch.load(best_model_path, map_location=device)
        
        if isinstance(ck, dict) and 'model_state_dict' in ck:
            model.load_state_dict(ck['model_state_dict'])
        else:
            model.load_state_dict(ck)
        print(f"Loaded model weights from {best_model_path}")
    model.to(device)
    model.eval()

    dl = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    metrics = []
    seq_idx = 0
    with torch.no_grad():
        for batch in dl:
            if seq_idx >= n_examples:
                break
            x = batch['x'].to(device)            # (1, T, in_dim)
            times = batch['times'].to(device)    # (1, T)
            pad_mask = batch['mask'].to(device)  # (1, T)
            y_gt = batch['y_gt'].to(device)      # (1, T, 2)
            gt_mask = batch['gt_mask'].to(device)# (1, T)
            # ensure mask shape for cfc
            pad_mask_exp = pad_mask.unsqueeze(-1)  # (1, T, 1)
            # forward
            pred = model.forward_sequence(x, times, pad_mask_exp)  # (1, T, 2)
            pred = pred.cpu().numpy()[0]  # (T,2)
            ygt = y_gt.cpu().numpy()[0]   # (T,2)
            times_np = times.cpu().numpy()[0]
            pad_mask_np = pad_mask.cpu().numpy()[0]
            gt_mask_np = gt_mask.cpu().numpy()[0]
            # reconstruct measurement points if present in inputs:
            # assume collate/input format uses gps_filled as input dims (for positions_only mode, x[:,:,0/1] are gps)
            x_np = x.cpu().numpy()[0]
            meas_x = x_np[:,0]  # gps_x filled (0 if missing)
            meas_y = x_np[:,1]
            gps_mask_np = batch.get('gps_mask', None)
            if gps_mask_np is not None:
                gps_mask_np = gps_mask_np.numpy()[0]
            else:
                # if not present, infer from padded zeros: where original pad_mask==1 and measurement != 0
                gps_mask_np = pad_mask_np * (np.abs(meas_x) + np.abs(meas_y) > 1e-6)

            # compute metrics only where gt_mask AND pad_mask ==1
            valid_mask = (gt_mask_np > 0.5) & (pad_mask_np > 0.5)
            if valid_mask.sum() == 0:
                # 没有 GT 则跳过但仍画图
                mse = [np.nan, np.nan]; mae = [np.nan, np.nan]; overall_rmse = np.nan
            else:
                diff = pred - ygt  # (T,2)
                mse_x = (diff[:,0]**2 * valid_mask).sum() / (valid_mask.sum() + 1e-8)
                mse_y = (diff[:,1]**2 * valid_mask).sum() / (valid_mask.sum() + 1e-8)
                mae_x = (np.abs(diff[:,0]) * valid_mask).sum() / (valid_mask.sum() + 1e-8)
                mae_y = (np.abs(diff[:,1]) * valid_mask).sum() / (valid_mask.sum() + 1e-8)
                overall_rmse = np.sqrt((mse_x + mse_y) / 2.0)
                mse = [np.sqrt(mse_x), np.sqrt(mse_y)]

            metrics.append({
                'seq': seq_idx,
                'rmse_x': float(mse[0]) if not np.isnan(mse[0]) else None,
                'rmse_y': float(mse[1]) if not np.isnan(mse[1]) else None,
                'mae_x': float(mae_x) if not np.isnan(mae_x) else None,
                'mae_y': float(mae_y) if not np.isnan(mae_y) else None,
                'overall_rmse': float(overall_rmse) if not np.isnan(overall_rmse) else None
            })

            # ---- plotting ----
            # 1) time-series plot for x and y
            fig, axes = plt.subplots(1,2, figsize=(figsize[0],figsize[1]))
            # X coordinate
            axes[0].plot(times_np, ygt[:,0], label='gt_x', linewidth=2)
            axes[0].plot(times_np, pred[:,0], '--', label='pred_x')
            axes[0].scatter(times_np[gps_mask_np>0.5], meas_x[gps_mask_np>0.5], s=10, c='gray', alpha=0.6, label='meas_x')
            axes[0].set_title(f'seq#{seq_idx} x vs time')
            axes[0].legend()
            # Y coordinate
            axes[1].plot(times_np, ygt[:,1], label='gt_y', linewidth=2)
            axes[1].plot(times_np, pred[:,1], '--', label='pred_y')
            axes[1].scatter(times_np[gps_mask_np>0.5], meas_y[gps_mask_np>0.5], s=10, c='gray', alpha=0.6, label='meas_y')
            axes[1].set_title(f'seq#{seq_idx} y vs time')
            axes[1].legend()
            plt.tight_layout()
            timeseries_path = os.path.join(out_dir, f'seq_{seq_idx:03d}_timeseries.png')
            plt.savefig(timeseries_path, bbox_inches='tight', dpi=150)
            plt.close(fig)

            # 2) XY trajectory plot
            fig2, ax2 = plt.subplots(1,1, figsize=(6,6))
            # plot GT segments (only where pad_mask)
            mask_idx = pad_mask_np > 0.5
            ax2.plot(ygt[mask_idx,0], ygt[mask_idx,1], '-', label='GT', linewidth=2)
            ax2.plot(pred[mask_idx,0], pred[mask_idx,1], '--', label='Pred', linewidth=2)
            # plot measurements as scatter (where gps mask true)
            ax2.scatter(meas_x[gps_mask_np>0.5], meas_y[gps_mask_np>0.5], s=10, c='orange', alpha=0.6, label='meas')
            ax2.set_title(f'seq#{seq_idx} XY trajectory (rmse={overall_rmse:.3f})')
            ax2.set_xlabel('x (m)'); ax2.set_ylabel('y (m)')
            ax2.legend(); ax2.axis('equal')
            traj_path = os.path.join(out_dir, f'seq_{seq_idx:03d}_traj.png')
            # plt.savefig(traj_path, bbox_inches='tight', dpi=150)
            plt.show()
            plt.close(fig2)

            # also save CSV of per-time predictions for this sequence
            rows = []
            for t in range(len(times_np)):
                if pad_mask_np[t] < 0.5: 
                    continue
                rows.append({
                    'seq': seq_idx,
                    'time': float(times_np[t]),
                    'meas_x': float(meas_x[t]) if gps_mask_np[t] > 0.5 else None,
                    'meas_y': float(meas_y[t]) if gps_mask_np[t] > 0.5 else None,
                    'gt_x': float(ygt[t,0]) if gt_mask_np[t] > 0.5 else None,
                    'gt_y': float(ygt[t,1]) if gt_mask_np[t] > 0.5 else None,
                    'pred_x': float(pred[t,0]),
                    'pred_y': float(pred[t,1]),
                    'has_gt': int(gt_mask_np[t])
                })
            df_seq = pd.DataFrame(rows)
            df_seq.to_csv(os.path.join(out_dir, f'seq_{seq_idx:03d}_preds.csv'), index=False)

            seq_idx += 1

    # summary
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(os.path.join(out_dir, 'val_metrics_summary.csv'), index=False)
    print("Saved visualizations & CSVs to", out_dir)
    # print overall stats
    valid_overall = df_metrics['overall_rmse'].dropna()
    if len(valid_overall) > 0:
        print("Validation overall RMSE mean:", float(valid_overall.mean()), "std:", float(valid_overall.std()))
    else:
        print("No GT available in the visualized sequences to compute overall RMSE.")
    return df_metrics


spec = importlib.util.spec_from_file_location("official_cfc", UPLOADED_CFC_PATH)
cfc_mod = importlib.util.module_from_spec(spec)
sys.modules["official_cfc"] = cfc_mod
spec.loader.exec_module(cfc_mod)
print("Loaded official CfC from", UPLOADED_CFC_PATH)

# -------------------------
# GPS-only augmentation utilities
# -------------------------
def add_gps_noise_and_spikes(times, gps, cfg, seed=None):
    """
    times: (N,) seconds
    gps: (N,2) gps_x,y with possible NaN (indicates missing)
    returns noisy_gps (N,2)
    """
    rng = np.random.RandomState(seed)
    N = len(times)
    dt = np.diff(times, prepend=times[0])
    gps_noisy = np.nan_to_num(gps.copy(), nan=0.0).astype(np.float32)

    # Gaussian noise
    gps_noisy += rng.randn(*gps_noisy.shape) * cfg['gps_sigma']

    # GPS bias random walk (slow drift)
    bias = np.zeros(2, dtype=np.float32)
    for i in range(N):
        step = rng.randn(2) * cfg['gps_bias_rw_sigma'] * math.sqrt(max(dt[i], 1e-6))
        bias += step
        gps_noisy[i] += bias

    # GPS spikes / jumps (multipath)
    total_time = times[-1] - times[0]
    num_spikes = rng.poisson(cfg['gps_spike_prob_per_s'] * max(total_time, 1.0))
    for _ in range(num_spikes):
        start_t = rng.uniform(times[0], times[-1])
        dur = rng.uniform(0.2, 2.0)
        amp = rng.uniform(cfg['gps_spike_amp'][0], cfg['gps_spike_amp'][1])
        inds = np.where((times >= start_t) & (times <= start_t + dur))[0]
        if len(inds) > 0:
            angle = rng.uniform(0, 2 * math.pi)
            offset = np.array([math.cos(angle), math.sin(angle)]) * amp
            gps_noisy[inds] += offset

    return gps_noisy

def interp_1d(orig_t, orig_v, target_t):
    orig_v = np.asarray(orig_v)
    # treat orig_v as NxD, use finite entries for interpolation per-dim
    if orig_v.ndim == 1:
        mask = np.isfinite(orig_v)
        if mask.sum() < 2:
            return np.full(len(target_t), np.nan, dtype=np.float32)
        return np.interp(target_t, orig_t[mask], orig_v[mask]).astype(np.float32)
    else:
        # per-dim
        D = orig_v.shape[1]
        out = np.stack([np.interp(target_t, orig_t[np.isfinite(orig_v[:,d])], orig_v[np.isfinite(orig_v[:,d]), d])
                        for d in range(D)], axis=1).astype(np.float32)
        return out

def upsample_sequence_to_hz(times, gps, gt, target_hz=100):
    t0, tN = float(times[0]), float(times[-1])
    up_dt = 1.0 / float(target_hz)
    up_times = np.arange(t0, tN + 1e-9, up_dt).astype(np.float32)
    up_gps = interp_1d(times, gps, up_times)
    up_gt = interp_1d(times, gt, up_times)
    # build gt_mask_up: mark where up_times align with original gt timestamps (within tol)
    orig_gt_mask = np.isfinite(gt[:,0]) & np.isfinite(gt[:,1])
    if orig_gt_mask.sum() == 0:
        gt_mask_up = np.zeros(len(up_times), dtype=np.float32)
    else:
        orig_gt_times = times[orig_gt_mask]
        tol = 0.5 / target_hz
        gt_mask_up = np.any(np.abs(up_times[:, None] - orig_gt_times[None, :]) <= tol, axis=1).astype(np.float32)
        # copy exact original GT values to matched indices
        for ot_idx, ot in enumerate(times):
            if not (np.isfinite(gt[ot_idx,0]) and np.isfinite(gt[ot_idx,1])):
                continue
            j = int(round((ot - t0) * target_hz))
            if 0 <= j < len(up_times):
                up_gt[j] = gt[ot_idx]
    return up_times, up_gps, up_gt, gt_mask_up

def upsample_and_augment_sequence(times, gps, gt, cfg, target_hz=100, seed=None):
    up_times, up_gps, up_gt, gt_mask_up = upsample_sequence_to_hz(times, gps, gt, target_hz=target_hz)
    gps_mask_up = np.isfinite(up_gps[:,0]) & np.isfinite(up_gps[:,1])
    gps_noisy = add_gps_noise_and_spikes(up_times, up_gps, cfg, seed=seed)
    x_up = gps_noisy.astype(np.float32)  # input features: [gps_x, gps_y]
    return x_up, up_times, up_gt, gt_mask_up, gps_mask_up

# -------------------------
# Dataset class (GPS-only)
# -------------------------
# class PositionsDataset(Dataset):
#     """
#     Each CSV must include: time,gps_x,gps_y,gt_x,gt_y
#     gt may have NaN at times where GT not provided; gps may have NaN where missing.
#     mode: '10hz' or 'upsample100'
#     """
#     def __init__(self, csv_paths, mode='10hz', upsample_hz=100, augment=True, cfg=AUG_CONFIG):
#         super().__init__()
#         self.sequences = []
#         self.mode = mode
#         self.upsample_hz = upsample_hz
#         self.augment = augment
#         self.cfg = cfg
#         self._load_csvs(csv_paths)

#     def _load_csvs(self, paths):
#         for p in paths:
#             df = pd.read_csv(p)
#             for required in ['time','gps_x','gps_y','gt_x','gt_y']:
#                 if COLUMN_MAP[required] not in df.columns:
#                     raise ValueError(f"CSV {p} missing required column: {COLUMN_MAP[required]}")
#             times = df[COLUMN_MAP['time']].values.astype(np.float32)
#             gps = np.stack([df[COLUMN_MAP['gps_x']].values, df[COLUMN_MAP['gps_y']].values], axis=1).astype(np.float32)
#             gt = np.stack([df[COLUMN_MAP['gt_x']].values, df[COLUMN_MAP['gt_y']].values], axis=1).astype(np.float32)
#             gt_mask = (~np.isnan(gt[:,0]) & ~np.isnan(gt[:,1])).astype(np.float32)
#             print("times shape is: ", times.shape)
#             print("gps shape is: ", gps.shape)
#             print("gt shape is: ", gt.shape)
#             print("gt_mask shape is: ", gt_mask.shape)
#             self.sequences.append({'times': times, 'gps': gps, 'gt': gt, 'gt_mask': gt_mask})

#     def __len__(self):
#         return len(self.sequences)

#     def __getitem__(self, idx):
#         s = self.sequences[idx]
#         times = s['times']; gps = s['gps']; gt = s['gt']; gt_mask = s['gt_mask']
#         if self.mode == '10hz':
#             if self.augment:
#                 gps_noisy = add_gps_noise_and_spikes(times, gps, self.cfg, seed=None)
#             else:
#                 gps_noisy = np.nan_to_num(gps.copy(), nan=0.0)
#             x = gps_noisy.astype(np.float32)  # (N,2)
#             gps_mask = (~np.isnan(gps[:,0]) & ~np.isnan(gps[:,1])).astype(np.float32)
#             return {'times': torch.from_numpy(times),
#                     'x': torch.from_numpy(x),
#                     'gps_mask': torch.from_numpy(gps_mask),
#                     'y_gt': torch.from_numpy(gt),
#                     'gt_mask': torch.from_numpy(gt_mask)}
#         else:
#             # upsample to target and augment inside
#             x_up, up_times, up_gt, gt_mask_up, gps_mask_up = upsample_and_augment_sequence(times, gps, gt, self.cfg, target_hz=self.upsample_hz, seed=None)
#             return {'times': torch.from_numpy(up_times),
#                     'x': torch.from_numpy(x_up),
#                     'gps_mask': torch.from_numpy(gps_mask_up.astype(np.float32)),
#                     'y_gt': torch.from_numpy(up_gt),
#                     'gt_mask': torch.from_numpy(gt_mask_up)}


# ---------- Replace existing PositionsDataset class with this improved one ----------
class PositionsDataset(Dataset):
    def __init__(self, csv_paths, mode='10hz', upsample_hz=100, augment=True, cfg=AUG_CONFIG,
                 seq_len_samples=50, stride_samples=20):     # the best_1 param(seq_len_samples=50, stride_samples=200)
        super().__init__()
        self.sequences = []  # 每个元素就是一个短序列 dict
        self.mode = mode
        self.upsample_hz = upsample_hz
        self.augment = augment
        self.cfg = cfg
        self.seq_len_samples = int(seq_len_samples)
        self.stride_samples = int(stride_samples)
        self._load_and_slice_csvs(csv_paths)

    def _load_and_slice_csvs(self, paths):
        for p in paths:
            df = pd.read_csv(p)
            # 校验列
            for required in ['time','gps_x','gps_y','gt_x','gt_y']:
                if COLUMN_MAP[required] not in df.columns:
                    raise ValueError(f"CSV {p} missing required column: {COLUMN_MAP[required]}")
            times_all = df[COLUMN_MAP['time']].values.astype(np.float32)
            gps_all = np.stack([df[COLUMN_MAP['gps_x']].values, df[COLUMN_MAP['gps_y']].values * 10], axis=1).astype(np.float32)
            gt_all = np.stack([df[COLUMN_MAP['gt_x']].values, df[COLUMN_MAP['gt_y']].values * 10], axis=1).astype(np.float32)
            # gps_all = np.stack([df[COLUMN_MAP['gps_x']].values, df[COLUMN_MAP['gps_y']].values], axis=1).astype(np.float32)
            # gt_all = np.stack([df[COLUMN_MAP['gt_x']].values, df[COLUMN_MAP['gt_y']].values], axis=1).astype(np.float32)
            gt_mask_all = (~np.isnan(gt_all[:,0]) & ~np.isnan(gt_all[:,1])).astype(np.float32)

            N = len(times_all)
            if N == 0:
                continue

            # If mode is upsample, we first upsample entire long sequence and then slice the result
            if self.mode == 'upsample100':
                up_times, up_gps, up_gt, gt_mask_up = upsample_sequence_to_hz(times_all, gps_all, gt_all, target_hz=self.upsample_hz)
                # now treat up_* as the full arrays to slice
                base_times = up_times
                base_gps = up_gps
                base_gt = up_gt
                base_gt_mask = gt_mask_up
                samples = len(base_times)
            else:
                base_times = times_all
                base_gps = gps_all
                base_gt = gt_all
                base_gt_mask = gt_mask_all
                samples = N

            # slice into windows of seq_len_samples with stride stride_samples
            seq_len = self.seq_len_samples
            stride = max(1, self.stride_samples)
            if samples < seq_len:
               
                start_indices = [0]
            else:
                start_indices = list(range(0, samples - seq_len + 1, stride))
                # include final tail window to cover end if not exactly divisible
                if (samples - seq_len) % stride != 0:
                    start_indices.append(samples - seq_len)

            for sidx in start_indices:
                eidx = sidx + seq_len
                seg_times = base_times[sidx:eidx]
                seg_gps = base_gps[sidx:eidx]
                seg_gt = base_gt[sidx:eidx]
                seg_gt_mask = base_gt_mask[sidx:eidx] if base_gt_mask is not None else np.zeros(seq_len, dtype=np.float32)
                # store raw segment (augmentation & upsample handling done in __getitem__)
                self.sequences.append({
                    'times': seg_times,
                    'gps': seg_gps,
                    'gt': seg_gt,
                    'gt_mask': seg_gt_mask
                })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        s = self.sequences[idx]
        times = s['times']; gps = s['gps']; gt = s['gt']; gt_mask = s['gt_mask']
        # mode == '10hz' : use provided slice directly; mode 'upsample100' was handled in slicing
        if self.augment:
            gps_noisy = add_gps_noise_and_spikes(times, gps, self.cfg, seed=None)
        else:
            gps_noisy = np.nan_to_num(gps.copy(), nan=0.0)
        x = gps_noisy.astype(np.float32)  # shape (seq_len, 2)
        gps_mask = (~np.isnan(gps[:,0]) & ~np.isnan(gps[:,1])).astype(np.float32)
        return {'times': torch.from_numpy(times),
                'x': torch.from_numpy(x),
                'gps_mask': torch.from_numpy(gps_mask),
                'y_gt': torch.from_numpy(gt),
                'gt_mask': torch.from_numpy(gt_mask)}
# ---------- end replacement ----------


def collate_fn(batch):
    B = len(batch)
    T = max(item['x'].shape[0] for item in batch)
    in_dim = batch[0]['x'].shape[1]  # should be 2
    x = torch.zeros(B, T, in_dim)
    times = torch.zeros(B, T)
    pad_mask = torch.zeros(B, T)
    gps_mask = torch.zeros(B, T)
    y_gt = torch.zeros(B, T, 2)
    gt_mask = torch.zeros(B, T)
    for i, item in enumerate(batch):
        L = item['x'].shape[0]
        x[i, :L] = item['x']
        times[i, :L] = item['times']
        pad_mask[i, :L] = 1.0
        gps_mask[i, :L] = item['gps_mask']
        y_gt[i, :L, :] = item['y_gt']
        gt_mask[i, :L] = item['gt_mask']
    return {'x': x, 'times': times, 'mask': pad_mask, 'gps_mask': gps_mask, 'y_gt': y_gt, 'gt_mask': gt_mask}

# -------------------------
# Model wrapper using official Cfc
# -------------------------
class OfficialCfCModel2DPos(nn.Module):
    def __init__(self, in_dim, hidden_dim, hparams):
        super().__init__()
        mask_extra = 1
        # output_dim = 2 (x,y)
        self.cfc = cfc_mod.Cfc(in_dim + mask_extra, hidden_dim, 2, hparams, return_sequences=True)
    def forward_sequence(self, x, times, mask):
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        pred = self.cfc.forward(x, times, mask)  # (B,T,2)
        return pred

def masked_mse_vec(pred, target, mask):
    mse = (pred - target)**2
    mse = mse.sum(dim=-1)
    return (mse * mask).sum() / (mask.sum() + 1e-8)

# -------------------------
# Train / eval
# -------------------------
def train(dataset, val_dataset=None, epochs=EPOCHS, device=DEVICE, mode='10hz'):
    dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    vdl = None if val_dataset is None else DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    hparams = {"backbone_activation": "gelu", "backbone_units": 128, "backbone_layers": 2}
    model = OfficialCfCModel2DPos(in_dim=2, hidden_dim=128, hparams=hparams).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_val = float('inf')
    best_path = '/home/rrc/xy_projects/LLN/weight_kitti/best_cfc_positions_10hz.pth' if mode=='10hz' else '/home/rrc/xy_projects/LLN/weight_kitti/best_cfc_positions_upsample100.pth'
    history = {'train_loss': [], 'val_loss': []}
    for ep in range(epochs):
        model.train()
        running = 0.0; c = 0
        for batch in dl:
            x = batch['x'].to(device); times = batch['times'].to(device)
            pad_mask = batch['mask'].to(device); y_gt = batch['y_gt'].to(device); gt_mask = batch['gt_mask'].to(device)
            effective_gt_mask = gt_mask * pad_mask
            pad_mask_exp = pad_mask.unsqueeze(-1)
            pred = model.forward_sequence(x, times, pad_mask_exp)  # (B,T,2)
            loss = masked_mse_vec(pred, y_gt, effective_gt_mask)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            running += loss.item() * x.size(0); c += x.size(0)
        train_loss = running / c; history['train_loss'].append(train_loss)
        val_loss = None
        if vdl is not None:
            model.eval(); vr = 0.0; vc = 0
            with torch.no_grad():
                for batch in vdl:
                    x = batch['x'].to(device); times = batch['times'].to(device)
                    pad_mask = batch['mask'].to(device); y_gt = batch['y_gt'].to(device); gt_mask = batch['gt_mask'].to(device)
                    effective_gt_mask = gt_mask * pad_mask
                    pad_mask_exp = pad_mask.unsqueeze(-1)
                    pred = model.forward_sequence(x, times, pad_mask_exp)
                    l = masked_mse_vec(pred, y_gt, effective_gt_mask)
                    vr += l.item() * x.size(0); vc += x.size(0)
            val_loss = vr / vc; history['val_loss'].append(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                torch.save({'model_state_dict': model.state_dict(), 'hparams': hparams}, best_path)
                print(f"New best saved to {best_path} (val_loss={val_loss:.6f})")
        print(f"Epoch {ep+1}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss if val_loss is not None else 'N/A'}")
    plt.figure(); plt.plot(history['train_loss'], label='train')
    if history['val_loss']: plt.plot(history['val_loss'], label='val')
    plt.legend(); plt.title('loss'); plt.show()
    return model, history

def export_predictions(model, dataset, out_csv='/home/rrc/xy_projects/LLN/weight_kitti/predictions_positions_only.csv', device=DEVICE):
    dl = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    model.eval(); rows=[]
    with torch.no_grad():
        for seq_idx, batch in enumerate(dl):
            x = batch['x'].to(device); times = batch['times'].to(device); pad_mask = batch['mask'].to(device)
            pad_mask_exp = pad_mask.unsqueeze(-1)
            pred = model.forward_sequence(x, times, pad_mask_exp).cpu().numpy()[0]
            ygt = batch['y_gt'].numpy(); gt_mask = batch['gt_mask'].numpy(); times_np = batch['times'].numpy(); mask_np = batch['mask'].numpy()
            for t in range(pred.shape[0]):
                if mask_np[0,t] < 0.5: continue
                rows.append({'seq': seq_idx, 't_idx': t, 'time': float(times_np[0,t]),
                             'gt_x': float(ygt[0,t,0]) if gt_mask[0,t]>0.5 else None,
                             'gt_y': float(ygt[0,t,1]) if gt_mask[0,t]>0.5 else None,
                             'pred_x': float(pred[t,0]), 'pred_y': float(pred[t,1]), 'has_gt': int(gt_mask[0,t])})
    df = pd.DataFrame(rows); df.to_csv(out_csv, index=False); print('Exported', out_csv); return df

# -------------------------
# Main
# -------------------------
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--mode', choices=['10hz','upsample100'], default='10hz')
#     parser.add_argument('--upsample_hz', type=int, default=100)
#     parser.add_argument('--epochs', type=int, default=EPOCHS)
#     parser.add_argument('--batch', type=int, default=BATCH_SIZE)
#     args = parser.parse_args()
#     if len(DATA_CSV_PATHS) == 0:
#         raise SystemExit("Please set DATA_CSV_PATHS in the script to point to your CSV files.")
#     # update globals
#     EPOCHS = args.epochs; BATCH_SIZE = args.batch
#     ds_all = PositionsDataset(DATA_CSV_PATHS, mode=args.mode, upsample_hz=args.upsample_hz, augment=True, cfg=AUG_CONFIG)
#     n = len(ds_all)
#     # val_n = max(1, int(0.15 * n))
#     print(f"Total sliced sequences loaded into dataset: {n}")
#     if n==0:
#         raise SystemExit("No sequences loaded.")
#     all_indices = list(range(n))
#     train_ds = torch.utils.data.Subset(ds_all, all_indices)
#     val_ds = train_ds
    
#     # print("time shape in squence is: ", ds_all['times'].shape)
#     # train_ds = torch.utils.data.Subset(ds_all, range(0, n - val_n))
#     # val_ds = torch.utils.data.Subset(ds_all, range(n - val_n, n))
#     model, history = train(train_ds, val_ds, epochs=EPOCHS, device=DEVICE, mode=args.mode)
#     export_predictions(model, val_ds, out_csv='/home/rrc/xy_projects/LLN/weight_kitti/predictions_positions_only.csv', device=DEVICE)
#     print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['10hz','upsample100'], default='10hz')
    parser.add_argument('--upsample_hz', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch', type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if len(DATA_CSV_PATHS) == 0:
        raise SystemExit("Please set DATA_CSV_PATHS in the script to point to your CSV files.")
    # update globals
    EPOCHS = args.epochs; BATCH_SIZE = args.batch
    ds_all = PositionsDataset(DATA_CSV_PATHS, mode=args.mode, upsample_hz=args.upsample_hz, augment=True, cfg=AUG_CONFIG)
    n = len(ds_all)
    # val_n = max(1, int(0.15 * n))
    print(f"Total sliced sequences loaded into dataset: {n}")
    if n==0:
        raise SystemExit("No sequences loaded.")
    all_indices = list(range(n))
    train_ds = torch.utils.data.Subset(ds_all, all_indices)
    val_ds = train_ds
    best_model = '/home/rrc/xy_projects/LLN/weight_kitti/best_cfc_positions_10hz_seq_02.pth'
    hparams = {"backbone_activation": "gelu", "backbone_units": 128, "backbone_layers": 2}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = OfficialCfCModel2DPos(in_dim=2, hidden_dim=128, hparams=hparams).to(device)
    ck = torch.load(best_model, map_location=device)
    model.load_state_dict(ck['model_state_dict'])
    # metrics_df = visualize_on_val(model, val_ds, device=device, out_dir='./kitti_weight/visualization', n_examples=6)
    full_val_ds = FullSequenceDataset(DATA_CSV_PATHS)
    df = plot_full_validation_trajectory(model, val_ds, device=device, best_model_path=best_model)