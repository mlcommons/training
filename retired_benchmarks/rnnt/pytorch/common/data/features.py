# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random

import librosa
import torch
import torch.nn as nn

from apex import amp


class BaseFeatures(nn.Module):
    """Base class for GPU accelerated audio preprocessing."""
    def __init__(self, optim_level):
        super(BaseFeatures, self).__init__()
        self.optim_level = optim_level

    @torch.no_grad()
    def calculate_features(self, audio, audio_lens):
        return audio, audio_lens

    def __call__(self, x):
        audio, audio_lens = x
        if self.optim_level == 1:
            with amp.disable_casts():
                return self.calculate_features(audio, audio_lens)
        else:
            return self.calculate_features(audio, audio_lens)


class SpecAugment(BaseFeatures):
    """Regularize by masking entire time steps/frequency bands.

    Implementes SpecAugment (https://arxiv.org/abs/1904.08779)
    with adaptive masking (https://arxiv.org/abs/1912.05533), without time
    warping.

    Args:
        freq_masks (int): number of masks for frequency bands
        min_freq (int): minimum number of frequencies in a single mask
        max_freq (int or float): maximum number of frequencies in a single mask
        time_masks (int or float): number of masks or adaptive percentage
        min_time (int): minimum number of masked time steps per mask; applies
            only if max is non-adaptive
        max_time (int or float): maximum number of masked time steps per mask,
            value 0 < 1 then denotes adaptive percentage
        noise_magnitude (float): mask with N(0, noise_magnitude * std(sample))
            noise instead of zeros to stabilize training
    """
    def __init__(self, optim_level, freq_masks=0, min_freq=0, max_freq=10, time_masks=0,
                 min_time=0, max_time=10, noise_magnitude=0):
        super(SpecAugment, self).__init__(optim_level)
        assert 0 <= min_freq <= max_freq
        assert 0 <= min_time <= max_time

        self.freq_masks = freq_masks
        self.min_freq = min_freq
        self.max_freq = max_freq

        self.time_masks = time_masks
        self.min_time = min_time
        self.max_time = max_time

        self.noise_magnitude = noise_magnitude

    @torch.no_grad()
    def calculate_features(self, x, x_lens):
        sh = x.shape
        mask = torch.zeros(x.shape, dtype=torch.bool, device=x.device)

        for idx in range(sh[0]):

            for _ in range(self.freq_masks):
                w = torch.randint(self.min_freq, self.max_freq + 1, size=(1,)).item()
                f0 = torch.randint(0, max(1, sh[1] - w + 1), size=(1,))
                mask[idx, f0:f0+w] = 1

            # Adaptive time masking
            time_masks = self.time_masks
            if 0 < time_masks < 1.0:
                time_masks = int(round(x_lens[idx].item() * time_masks))

            max_time = self.max_time
            if 0 < max_time < 1.0:
                max_time = int(round(x_lens[idx].item() * max_time))

            for _ in range(time_masks):
                w = torch.randint(self.min_time, max_time + 1, size=(1,)).item()
                t0 = torch.randint(0, max(1, sh[2] - w + 1), size=(1,))
                mask[idx, :, t0:t0+w] = 1

        if self.noise_magnitude > 0:
            mean = torch.zeros(x.size(0), x.size(1), 1, device=x.device)
            std = torch.zeros(x.size(0), x.size(1), 1, device=x.device)
            for idx in range(sh[0]):
                mean[idx, :, 0] = x[idx, :, :x_lens[idx]].mean(dim=1)
                std[idx, :, 0] = x[idx, :, :x_lens[idx]].mean(dim=1)

            std *= self.noise_magnitude
            noise = (mean + torch.randn_like(x) * std).masked_fill(~mask, 0)
        else:
            noise = 0

        return x.masked_fill(mask, 0) + noise, x_lens


@torch.jit.script
def normalize_batch(x, x_lens, normalize_type: str):
    if normalize_type == "per_feature":
        mean = x.new_zeros(x.size(0), x.size(1))
        std = x.new_zeros(x.size(0), x.size(1))

        for i in range(x.size(0)):
            mean[i, :] = x[i, :, :x_lens[i]].mean(dim=1)
            std[i, :] = x[i, :, :x_lens[i]].std(dim=1)
        # make sure std is not zero
        return (x - mean.unsqueeze(2)) / (std.unsqueeze(2) + 1e-5)

    elif normalize_type == "all_features":
        mean = x.new_zeros(x.size(0))
        std = x.new_zeros(x.size(0))
        for i in range(x.size(0)):
            mean[i] = x[i, :, :x_lens[i]].mean()
            std[i] = x[i, :, :x_lens[i]].std()
        # make sure x_std is not zero
        return (x - mean.view(-1, 1, 1)) / (std.view(-1, 1, 1) + 1e-5)
    else:
        return x


def stack_subsample_frames(x, x_lens, stacking=1, subsampling=1):
    """ Stacks frames together across feature dim, and then subsamples

    input is batch_size, feature_dim, num_frames
    output is batch_size, feature_dim * stacking, num_frames / subsampling

    """
    seq = [x]
    for n in range(1, stacking):
        tmp = torch.zeros_like(x)
        tmp[:, :, :-n] = x[:, :, n:]
        seq.append(tmp)
    x = torch.cat(seq, dim=1)[:, :, ::subsampling]

    if subsampling > 1:
        x_lens = torch.ceil(x_lens.float() / subsampling).int()

        if x.size(2) > x_lens.max().item():
            assert abs(x.size(2) - x_lens.max().item()) <= 1
            x = x[:,:,:x_lens.max().item()]

    return x, x_lens


class FilterbankFeatures(BaseFeatures):
    # For JIT, https://pytorch.org/docs/stable/jit.html#python-defined-constants
    __constants__ = ["dither", "preemph", "n_fft", "hop_length", "win_length",
                     "log", "normalize"]
    # torchscript: "center" removed due to a bug

    def __init__(self,
                 optim_level, sample_rate=8000, window_size=0.02, window_stride=0.01,
                 window="hamming", normalize="per_feature", n_fft=None,
                 preemph=0.97, n_filt=64, lowfreq=0, highfreq=None, log=True,
                 dither=1e-5):
        super(FilterbankFeatures, self).__init__(optim_level)
        torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'none': None,
        }

        self.win_length = int(sample_rate * window_size) # frame size
        self.hop_length = int(sample_rate * window_stride)
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))

        self.normalize = normalize
        self.log = log
        #TORCHSCRIPT: Check whether or not we need this
        self.dither = dither
        self.n_filt = n_filt
        self.preemph = preemph
        highfreq = highfreq or sample_rate / 2
        window_fn = torch_windows.get(window, None)
        window_tensor = window_fn(self.win_length,
                                  periodic=False) if window_fn else None
        filterbanks = torch.tensor(
            librosa.filters.mel(sample_rate, self.n_fft, n_mels=n_filt,
                                fmin=lowfreq, fmax=highfreq),
            dtype=torch.float).unsqueeze(0)
        # torchscript
        self.register_buffer("fb", filterbanks)
        self.register_buffer("window", window_tensor)

    # do stft
    # TORCHSCRIPT: center removed due to bug
    def stft(self, x):
        return torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.win_length,
                          window=self.window.to(dtype=torch.float))
                          # return_complex=False)


    @torch.no_grad()
    def calculate_features(self, x, x_lens):
        if self.dither > 0:
            x += self.dither * torch.randn_like(x)

        if self.preemph is not None:
            x = torch.cat(
                (x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1)
        x  = self.stft(x).to(x.dtype)

        x_lens = torch.ceil(x_lens.float() / self.hop_length).int()

        # get power spectrum
        x = x.pow(2).sum(-1)

        # dot with filterbank energies
        x = torch.matmul(self.fb.to(x.dtype), x)

        if self.log:
            x = torch.log(x + 1e-20)

        # normalize if required
        x = normalize_batch(x, x_lens, normalize_type=self.normalize)

        return x, x_lens

class FrameSplicing(BaseFeatures):
    __constants__ = ['frame_subsampling', 'frame_stacking']

    def __init__(self, optim_level, frame_stacking=1, frame_subsampling=1):
        super(FrameSplicing, self).__init__(optim_level)
        self.frame_stacking = frame_stacking
        self.frame_subsampling = frame_subsampling

    def calculate_features(self, x, x_lens):

        # frame splicing if required
        if self.frame_stacking > 1 or self.frame_subsampling > 1:
            x, x_lens = stack_subsample_frames(x, x_lens, self.frame_stacking,
                                               self.frame_subsampling)

        return x, x_lens

class FillPadding(BaseFeatures):
    __constants__ = [ 'fill_value' ]
    def __init__(self, optim_level, fill_value=0):
        super(FillPadding, self).__init__(optim_level)
        self.fill_value = fill_value

    def calculate_features(self, x, x_lens):
        # mask to zero any values beyond x_lens in batch,
        max_len = x.size(-1)
        mask = torch.arange(max_len, dtype=x_lens.dtype, device=x.device)
        mask = mask.expand(x.size(0), max_len) >= x_lens.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1), self.fill_value)

        return x, x_lens

