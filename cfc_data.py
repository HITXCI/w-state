import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class IrregularVehicleMultiSensorDataset(Dataset):
    def __init__(self, n_sequences=1000, seq_len=80, dt_mean=0.1, seed=0):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.sequences = []
        for _ in range(n_sequences):
            t = [0.0]
            pos = [0.0]
            velocity = 5.0 + rng.randn()*0.5
            velocities = [velocity]
            acc = 0.0
            for i in range(seq_len-1):
                dt = max(1e-3, rng.normal(dt_mean, dt_mean*0.4))
                t.append(t[-1] + dt)
                if rng.rand() < 0.15:
                    acc = rng.randn()*1.5
                velocity = velocity + acc*dt
                pos.append(pos[-1] + velocity*dt)
                velocities.append(velocity)
            velocities = np.array(velocities)
            times = np.array(t)
            dt_array = np.diff(times, prepend=times[0])
            acc_true = np.concatenate([[0.0], np.diff(velocities) / (dt_array[1:]+1e-8)])
            # sensors: IMU acc with bias + noise, wheel speed (noisy, slightly biased), GPS position sampled sparsely
            imu_acc = acc_true + rng.randn(*acc_true.shape)*0.05 + 0.02  # small bias
            wheel_speed = velocities + rng.randn(*velocities.shape)*0.1 + 0.01
            # sparse GPS: sampled every ~1s with larger noise; fill with nan where unavailable
            gps_pos = np.full_like(velocities, np.nan)
            gps_interval = max(1, int(round(1.0 / dt_mean)))
            for idx in range(0, len(velocities), gps_interval):
                gps_pos[idx] = pos[idx] + rng.randn()*0.5

            seq = {
                'times': times.astype(np.float32),
                'imu_acc': imu_acc.astype(np.float32),
                'wheel_speed': wheel_speed.astype(np.float32),
                'gps_pos': gps_pos.astype(np.float32),
                'vel_true': velocities.astype(np.float32),
                'pos_true': np.array(pos).astype(np.float32)
            }
            self.sequences.append(seq)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        s = self.sequences[idx]
        # Input channels: wheel_speed, imu_acc, gps_pos (nan->0 and mask)
        gps_filled = np.nan_to_num(s['gps_pos'], nan=0.0)
        x = np.stack([s['wheel_speed'], s['imu_acc'], gps_filled], axis=1)  # (T, 3)
        gps_mask = (~np.isnan(s['gps_pos'])).astype(np.float32)
        return {
            'times': torch.from_numpy(s['times']),
            'x': torch.from_numpy(x),
            'gps_mask': torch.from_numpy(gps_mask),
            'y_vel': torch.from_numpy(s['vel_true']),
            'y_pos': torch.from_numpy(s['pos_true'])
        }

# collate that pads sequences to the same length in a batch
def collate_fn(batch):
    batch_size = len(batch)
    T = max(item['x'].shape[0] for item in batch)
    in_dim = batch[0]['x'].shape[1]
    x = torch.zeros(batch_size, T, in_dim)
    y_vel = torch.zeros(batch_size, T)
    y_pos = torch.zeros(batch_size, T)
    times = torch.zeros(batch_size, T)
    mask = torch.zeros(batch_size, T)
    gps_mask = torch.zeros(batch_size, T)
    for i, item in enumerate(batch):
        L = item['x'].shape[0]
        x[i, :L] = item['x']
        y_vel[i, :L] = item['y_vel']
        y_pos[i, :L] = item['y_pos']
        times[i, :L] = item['times']
        mask[i, :L] = 1.0
        gps_mask[i, :L] = item['gps_mask']
    return {'x': x, 'y_vel': y_vel, 'y_pos': y_pos, 'times': times, 'mask': mask, 'gps_mask': gps_mask}

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_map = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.recurrent_map = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for w in self.input_map.parameters():
            if w.dim() == 1:
                torch.nn.init.uniform_(w, -0.1, 0.1)
            else:
                torch.nn.init.xavier_uniform_(w)
        for w in self.recurrent_map.parameters():
            if w.dim() == 1:
                torch.nn.init.uniform_(w, -0.1, 0.1)
            else:
                torch.nn.init.orthogonal_(w)

    def forward(self, inputs, states):
        output_state, cell_state = states

        z = self.input_map(inputs) + self.recurrent_map(output_state)
        i, ig, fg, og = z.chunk(4, 1)

        input_activation = self.tanh(i)
        input_gate = self.sigmoid(ig)
        forget_gate = self.sigmoid(fg + 1.0)
        output_gate = self.sigmoid(og)

        new_cell = cell_state * forget_gate + input_activation * input_gate
        output_state = self.tanh(new_cell) * output_gate

        return output_state, new_cell


class LeCun(nn.Module):
    def __init__(self):
        super(LeCun, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)


class CfcCell(nn.Module):
    def __init__(self, input_size, hidden_size, hparams):
        super(CfcCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hparams = hparams
        self._no_gate = False
        if "no_gate" in self.hparams:
            self._no_gate = self.hparams["no_gate"]
        self._minimal = False
        if "minimal" in self.hparams:
            self._minimal = self.hparams["minimal"]

        if self.hparams["backbone_activation"] == "silu":
            backbone_activation = nn.SiLU
        elif self.hparams["backbone_activation"] == "relu":
            backbone_activation = nn.ReLU
        elif self.hparams["backbone_activation"] == "tanh":
            backbone_activation = nn.Tanh
        elif self.hparams["backbone_activation"] == "gelu":
            backbone_activation = nn.GELU
        elif self.hparams["backbone_activation"] == "lecun":
            backbone_activation = LeCun
        else:
            raise ValueError("Unknown activation")
        layer_list = [
            nn.Linear(input_size + hidden_size, self.hparams["backbone_units"]),
            backbone_activation(),
        ]
        for i in range(1, self.hparams["backbone_layers"]):
            layer_list.append(
                nn.Linear(
                    self.hparams["backbone_units"], self.hparams["backbone_units"]
                )
            )
            layer_list.append(backbone_activation())
            if "backbone_dr" in self.hparams.keys():
                layer_list.append(torch.nn.Dropout(self.hparams["backbone_dr"]))
        self.backbone = nn.Sequential(*layer_list)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.ff1 = nn.Linear(self.hparams["backbone_units"], hidden_size)
        if self._minimal:
            self.w_tau = torch.nn.Parameter(
                data=torch.zeros(1, self.hidden_size), requires_grad=True
            )
            self.A = torch.nn.Parameter(
                data=torch.ones(1, self.hidden_size), requires_grad=True
            )
        else:
            self.ff2 = nn.Linear(self.hparams["backbone_units"], hidden_size)
            self.time_a = nn.Linear(self.hparams["backbone_units"], hidden_size)
            self.time_b = nn.Linear(self.hparams["backbone_units"], hidden_size)
        self.init_weights()

    def init_weights(self):
        init_gain = self.hparams.get("init")
        if init_gain is not None:
            for w in self.parameters():
                if w.dim() == 2:
                    torch.nn.init.xavier_uniform_(w, gain=init_gain)

    def forward(self, input, hx, ts):

        batch_size = input.size(0)
        ts = ts.view(batch_size, 1)
        x = torch.cat([input, hx], 1)

        x = self.backbone(x)
        if self._minimal:
            # Solution
            ff1 = self.ff1(x)
            new_hidden = (
                -self.A
                * torch.exp(-ts * (torch.abs(self.w_tau) + torch.abs(ff1)))
                * ff1
                + self.A
            )
        else:
            # Cfc
            ff1 = self.tanh(self.ff1(x))
            ff2 = self.tanh(self.ff2(x))
            t_a = self.time_a(x)
            t_b = self.time_b(x)
            t_interp = self.sigmoid(t_a * ts + t_b)
            if self._no_gate:
                new_hidden = ff1 + t_interp * ff2
            else:
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2
        return new_hidden



class Cfc(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_size,
        out_feature_vel,
        out_feature_pos,
        hparams,
        return_sequences=False,
        use_mixed=False,
        use_ltc=False,
    ):
        super(Cfc, self).__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_feature_vel = out_feature_vel
        self.out_feature_pos = out_feature_pos
        self.return_sequences = return_sequences

        if use_ltc:
            self.rnn_cell = LTCCell(in_features, hidden_size)
        else:
            self.rnn_cell = CfcCell(in_features, hidden_size, hparams)
        self.use_mixed = use_mixed
        if self.use_mixed:
            self.lstm = LSTMCell(in_features, hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.out_feature_vel)
        self.fc_pos = nn.Linear(self.hidden_size, self.out_feature_pos)

    def forward_sequence(self, x, timespans=None, mask=None):
        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)
        true_in_features = x.size(2)
        h_state = torch.zeros((batch_size, self.hidden_size), device=device)
        if self.use_mixed:
            c_state = torch.zeros((batch_size, self.hidden_size), device=device)
        output_sequence = []
        output_sequence_pos = []
        if mask is not None:
            forwarded_output = torch.zeros(
                (batch_size, self.out_feature_vel), device=device
            )
            forwarded_input = torch.zeros((batch_size, true_in_features), device=device)
            time_since_update = torch.zeros(
                (batch_size, true_in_features), device=device
            )
        for t in range(seq_len):
            inputs = x[:, t]
            ts = timespans[:, t].squeeze()
            if mask is not None:
                if mask.size(-1) == true_in_features:
                    forwarded_input = (
                        mask[:, t] * inputs + (1 - mask[:, t]) * forwarded_input
                    )
                    time_since_update = (ts.view(batch_size, 1) + time_since_update) * (
                        1 - mask[:, t]
                    )
                else:
                    forwarded_input = inputs
                if (
                    true_in_features * 2 < self.in_features
                    and mask.size(-1) == true_in_features
                ):
                    # we have 3x in-features
                    inputs = torch.cat(
                        (forwarded_input, time_since_update, mask[:, t]), dim=1
                    )
                else:
                    # we have 2x in-feature
                    inputs = torch.cat((forwarded_input, mask[:, t]), dim=1)
            if self.use_mixed:
                h_state, c_state = self.lstm(inputs, (h_state, c_state))
            h_state = self.rnn_cell.forward(inputs, h_state, ts)
            if mask is not None:
                cur_mask, _ = torch.max(mask[:, t], dim=1)
                cur_mask = cur_mask.view(batch_size, 1)
                current_output = self.fc(h_state)
                forwarded_output = (
                    cur_mask * current_output + (1.0 - cur_mask) * forwarded_output
                )
            if self.return_sequences:
                output_sequence.append(self.fc(h_state))
                output_sequence_pos.append(self.fc_pos(h_state))

        if self.return_sequences:
            readout = torch.stack(output_sequence, dim=1)
            readout_pos = torch.stack(output_sequence_pos, dim=1)
        elif mask is not None:
            readout = forwarded_output
            readout_pos = forwarded_output
        else:
            readout = self.fc(h_state)
            readout_pos = self.fc_pos(h_state)
        return readout, readout_pos


class LTCCell(nn.Module):
    def __init__(
        self,
        in_features,
        units,
        ode_unfolds=6,
        epsilon=1e-8,
    ):
        super(LTCCell, self).__init__()
        self.in_features = in_features
        self.units = units
        self._init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        # self.softplus = nn.Softplus()
        self.softplus = nn.Identity()
        self._allocate_parameters()

    @property
    def state_size(self):
        return self.units

    @property
    def sensory_size(self):
        return self.in_features

    def add_weight(self, name, init_value):
        param = torch.nn.Parameter(init_value)
        self.register_parameter(name, param)
        return param

    def _get_init_value(self, shape, param_name):
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return torch.ones(shape) * minval
        else:
            return torch.rand(*shape) * (maxval - minval) + minval

    def _erev_initializer(self, shape=None):
        return np.random.default_rng().choice([-1, 1], size=shape)

    def _allocate_parameters(self):
        self._params = {}
        self._params["gleak"] = self.add_weight(
            name="gleak", init_value=self._get_init_value((self.state_size,), "gleak")
        )
        self._params["vleak"] = self.add_weight(
            name="vleak", init_value=self._get_init_value((self.state_size,), "vleak")
        )
        self._params["cm"] = self.add_weight(
            name="cm", init_value=self._get_init_value((self.state_size,), "cm")
        )
        self._params["sigma"] = self.add_weight(
            name="sigma",
            init_value=self._get_init_value(
                (self.state_size, self.state_size), "sigma"
            ),
        )
        self._params["mu"] = self.add_weight(
            name="mu",
            init_value=self._get_init_value((self.state_size, self.state_size), "mu"),
        )
        self._params["w"] = self.add_weight(
            name="w",
            init_value=self._get_init_value((self.state_size, self.state_size), "w"),
        )
        self._params["erev"] = self.add_weight(
            name="erev",
            init_value=torch.Tensor(
                self._erev_initializer((self.state_size, self.state_size))
            ),
        )
        self._params["sensory_sigma"] = self.add_weight(
            name="sensory_sigma",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_sigma"
            ),
        )
        self._params["sensory_mu"] = self.add_weight(
            name="sensory_mu",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_mu"
            ),
        )
        self._params["sensory_w"] = self.add_weight(
            name="sensory_w",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_w"
            ),
        )
        self._params["sensory_erev"] = self.add_weight(
            name="sensory_erev",
            init_value=torch.Tensor(
                self._erev_initializer((self.sensory_size, self.state_size))
            ),
        )

        self._params["input_w"] = self.add_weight(
            name="input_w",
            init_value=torch.ones((self.sensory_size,)),
        )
        self._params["input_b"] = self.add_weight(
            name="input_b",
            init_value=torch.zeros((self.sensory_size,)),
        )


    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = torch.unsqueeze(v_pre, -1)  # For broadcasting
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def _ode_solver(self, inputs, state, elapsed_time):
        v_pre = state

        
        sensory_w_activation = self.softplus(self._params["sensory_w"]) * self._sigmoid(
            inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        )

        sensory_rev_activation = sensory_w_activation * self._params["sensory_erev"]

        # Reduce over dimension 1 (=source sensory neurons)
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        # cm/t is loop invariant
        cm_t = self.softplus(self._params["cm"]).view(1, -1) / (
            (elapsed_time + 1) / self._ode_unfolds
        )

        # Unfold the multiply ODE multiple times into one RNN step
        for t in range(self._ode_unfolds):
            w_activation = self.softplus(self._params["w"]) * self._sigmoid(
                v_pre, self._params["mu"], self._params["sigma"]
            )

            rev_activation = w_activation * self._params["erev"]

            # Reduce over dimension 1 (=source neurons)
            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory

            numerator = (
                cm_t * v_pre
                + self.softplus(self._params["gleak"]) * self._params["vleak"]
                + w_numerator
            )
            denominator = cm_t + self.softplus(self._params["gleak"]) + w_denominator

            # Avoid dividing by 0
            v_pre = numerator / (denominator + self._epsilon)
            if torch.any(torch.isnan(v_pre)):
                breakpoint()
        return v_pre

    def _map_inputs(self, inputs):
        inputs = inputs * self._params["input_w"]
        inputs = inputs + self._params["input_b"]
        return inputs

    def _map_outputs(self, state):
        output = state
        output = output * self._params["output_w"]
        output = output + self._params["output_b"]
        return output

    def _clip(self, w):
        return torch.nn.ReLU()(w)

    def apply_weight_constraints(self):
        self._params["w"].data = self._clip(self._params["w"].data)
        self._params["sensory_w"].data = self._clip(self._params["sensory_w"].data)
        self._params["cm"].data = self._clip(self._params["cm"].data)
        self._params["gleak"].data = self._clip(self._params["gleak"].data)

    def forward(self, input, hx, ts):
        # Regularly sampled mode (elapsed time = 1 second)
        ts = ts.view((-1, 1))
        inputs = self._map_inputs(input)

        next_state = self._ode_solver(inputs, hx, ts)

        # outputs = self._map_outputs(next_state)

        return next_state

def masked_mse(pred, target, mask):
    mse = (pred - target)**2
    return (mse * mask).sum() / (mask.sum() + 1e-8)

def masked_mae(pred, target, mask):
    mae = torch.abs(pred - target)
    return (mae * mask).sum() / (mask.sum() + 1e-8)

# -------------------------
# Training / Eval / Save / Export
# -------------------------

def train_and_save(epochs=25, device='cpu'):
    ds = IrregularVehicleMultiSensorDataset(n_sequences=900, seq_len=70)
    val_ds = IrregularVehicleMultiSensorDataset(n_sequences=150, seq_len=70, seed=1234)
    dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    vdl = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    hparams = {
    "epochs": 20,
    "class_weight": 11.69,
    "clipnorm": 0,
    "hidden_size": 256,
    "base_lr": 0.002,
    "decay_lr": 0.9,
    "backbone_activation": "silu",
    "backbone_units": 64,
    "backbone_dr": 0.2,
    "backbone_layers": 2,
    "weight_decay": 4e-06,
    "optim": "adamw",
    "init": 0.5,
    # "batch_size": 128,
    "use_mixed": False,
    "no_gate": False,
    "minimal": False,
    "use_ltc": False,
}

    model = Cfc(in_features=3, hidden_size=128, out_feature_vel=1, out_feature_pos=1, hparams=hparams).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    # composite loss: velocity MSE + position MSE (weights adjustable)
    vel_w = 1.0
    pos_w = 1.0

    history = {'train_loss': [], 'val_loss': [], 'train_rmse_vel': [], 'val_rmse_vel': []}

    best_val_loss = float('inf')

    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        count = 0
        for batch in dl:
            x = batch['x'].to(device)
            y_vel = batch['y_vel'].to(device)
            y_pos = batch['y_pos'].to(device)
            times = batch['times'].to(device)
            mask = batch['mask'].to(device)

            pred_vel, pred_pos = model.forward_sequence(x, times, mask)
            loss_vel = masked_mse(pred_vel, y_vel, mask)
            loss_pos = masked_mse(pred_pos, y_pos, mask)
            loss = vel_w * loss_vel + pos_w * loss_pos

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            running_loss += loss.item() * x.size(0)
            count += x.size(0)
        train_loss = running_loss / count

        # validation
        model.eval()
        vrunning = 0.0
        vcount = 0
        vel_rmse = 0.0
        with torch.no_grad():
            for batch in vdl:
                x = batch['x'].to(device)
                y_vel = batch['y_vel'].to(device)
                y_pos = batch['y_pos'].to(device)
                times = batch['times'].to(device)
                mask = batch['mask'].to(device)
                pred_vel, pred_pos = model.forward_sequence(x, times, mask)
                loss_vel = masked_mse(pred_vel, y_vel, mask)
                loss_pos = masked_mse(pred_pos, y_pos, mask)
                loss = vel_w * loss_vel + pos_w * loss_pos
                vrunning += loss.item() * x.size(0)
                vcount += x.size(0)
                # RMSE vel per batch
                vel_rmse += (torch.sqrt(masked_mse(pred_vel, y_vel, mask)).item()) * x.size(0)
        val_loss = vrunning / vcount
        val_rmse_vel = vel_rmse / vcount

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse_vel'].append(val_rmse_vel)

        print(f"Epoch {ep+1}/{epochs}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  val_rmse_vel={val_rmse_vel:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dir = './data_xy'
            best_model_path = os.path.join(save_dir, 'cfc_vehicle_model.pth')
            torch.save(model.state_dict(), best_model_path)

    # plot losses
    plt.figure(figsize=(6,4))
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Training/Validation Loss')
    plt.show()

    # pick a validation batch for visualization + export
    batch = next(iter(vdl))
    x = batch['x'].to(device)
    y_vel = batch['y_vel'].to(device)
    y_pos = batch['y_pos'].to(device)
    times = batch['times'].to(device)
    mask = batch['mask'].to(device)
    with torch.no_grad():
        pred_vel, pred_pos = model.forward_sequence(x, times, mask)

    pred_vel = pred_vel.cpu().numpy()
    pred_pos = pred_pos.cpu().numpy()
    y_vel = y_vel.cpu().numpy()
    y_pos = y_pos.cpu().numpy()
    times = times.cpu().numpy()
    mask = mask.cpu().numpy()

    # plot first 6 sequences
    nplot = min(6, pred_vel.shape[0])
    fig, axs = plt.subplots(nplot, 2, figsize=(12, 2.5*nplot))
    for i in range(nplot):
        L = int(mask[i].sum())
        axs[i,0].plot(times[i,:L], y_vel[i,:L], label='gt_vel')
        axs[i,0].plot(times[i,:L], pred_vel[i,:L], label='pred_vel')
        axs[i,0].set_ylabel('velocity')
        axs[i,0].legend()
        axs[i,1].plot(times[i,:L], y_pos[i,:L], label='gt_pos')
        axs[i,1].plot(times[i,:L], pred_pos[i,:L], label='pred_pos')
        axs[i,1].set_ylabel('position')
        axs[i,1].legend()
    axs[-1,0].set_xlabel('time (s)')
    axs[-1,1].set_xlabel('time (s)')
    plt.tight_layout()
    plt.show()

    # Save model
    # save_dir = './data'
    # os.makedirs(save_dir, exist_ok=True)
    # model_path = os.path.join(save_dir, 'cfc_vehicle_model.pth')
    # torch.save({'model_state_dict': model.state_dict()}, model_path)
    # print('Saved model to', model_path)

    # Export predictions to CSV (first batch only)
    rows = []
    B, T = pred_vel.shape
    for i in range(B):
        L = int(mask[i].sum())
        for t in range(L):
            rows.append({
                'seq': i,
                't_idx': t,
                'time': float(times[i,t]),
                'gt_vel': float(y_vel[i,t]),
                'pred_vel': float(pred_vel[i,t]),
                'gt_pos': float(y_pos[i,t]),
                'pred_pos': float(pred_pos[i,t])
            })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, 'predictions.csv')
    df.to_csv(csv_path, index=False)
    print('Exported predictions to', csv_path)

    return model, history, df

# -------------------------
# If run as script
# -------------------------
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)
    model, history, df = train_and_save(epochs=20, device=device)
    print('Done.')
