'''
Author: Xiaohan Chen
Email: cxh_bb@outlook.com
'''

import numpy as np
import torch
import torchaudio
from typing import Optional
import time

def probability():
    return np.random.random()

def ToTensor(signal):
    return torch.from_numpy(signal).float()

def exponential(signal: torch.tensor):
    return 5. * torch.exp(-5.*signal)

def FFT(signal: torch.tensor):
    '''
    NOTE: input waveform signal
    '''
    assert len(signal.size()) == 2
    signal = torch.fft.fft(signal)
    signal = 2 * torch.abs(signal) / len(signal)
    signal = signal[:,:int(signal.size(-1) / 2)]
    signal[:,0] = 0.
    return signal

def STFT(signal: torch.tensor, n_fft: int = 400, power: Optional[float] = 2.0):
    '''
    NOTE: input waveform signal
    '''
    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        power=power,
    )
    return spectrogram(signal)


# ==============================================
#             Waveform augmentations
# ==============================================
def AddGaussianSNR(signal: torch.tensor, min_snr: int, max_snr: int):
    '''
    Adapted from: https://github.com/iver56/audiomentations/blob/master/audiomentations/augmentations/add_gaussian_snr.py
    Add Gaussian noise to the signal, a random signal-to-noise ratio will be picked.
    '''
    # min_snr = 0,  # (int): minimum signal-to-noise ration in dB
    # max_snr = 30, # (int): maximum signal-to-noise ration in dB
    signal_length = signal.size(-1)
    signal_copy = signal.clone()
    device = signal.device

    snr = np.random.randint(min_snr, max_snr, dtype=int)

    clear_rms = torch.sqrt(torch.mean(torch.square(signal)))
    a = float(snr) / 20
    noise_rms = clear_rms / (10**a)
    noise = torch.normal(0.0, noise_rms, size=(signal_length,))

    return signal_copy + noise.to(device)

def Shift(signal: torch.tensor):
    '''
    Shift the signal forwards or backwards
    '''
    shift_factor = np.random.uniform(0,0.5)
    shift_length = round(signal.size(-1) * shift_factor)
    num_places_to_shift = round(np.random.uniform(-shift_length, shift_length))
    shifted_signal = torch.roll(signal, num_places_to_shift, dims=-1)
    return shifted_signal

def TimeMask(signal: torch.tensor):
    '''
    Randomly mask a part of signal
    '''
    signal_length = signal.size(-1)
    device = signal.device
    signal_copy = signal.clone()
    fill_value = 0.

    mask_factor = np.random.uniform(0.1,0.5)
    max_mask_length = int(signal_length * mask_factor)  # maximum mask band length
    mask_length = np.random.randint(max_mask_length) # randomly choose a mask band length
    mask_start = np.random.randint(0, signal_length) # randomly choose a mask band start point

    while mask_start + mask_length > max_mask_length:
        mask_start = np.random.randint(0, signal_length)
    mask = torch.arange(signal_length, device=device)
    mask = (mask >= mask_start) & (mask < mask_start + mask_length)
    
    signal_copy = signal_copy.masked_fill(mask, fill_value)
    
    return signal_copy

def Gain(signal: torch.tensor):
    '''
    Adapted from: https://pytorch.org/audio/stable/generated/torchaudio.transforms.Vol.html#torchaudio.transforms.Vol
    Multiply the audio by a random amplitude factor to reduce or increase the volume.
    '''
    gain_min = 0.5
    gain_max = 1.5
    gain_factor = np.random.uniform(gain_min, gain_max)

    signal_copy = signal.clone()

    return signal_copy * gain_factor

def Flip(signal: torch.tensor):
    '''
    Time inversion
    Horizontally reverse the signal: start <--> end
    '''
    if len(signal.size()) == 1:
        signal_copy = torch.flip(signal, dims=[0])
    elif len(signal.size()) == 2:
        signal_copy = torch.flip(signal, dims=[1])
    else:
        raise Exception(f"{signal.size()} dimentinal signal time masking is not implemented, "
                        "please try 1 or 2 dimentional signal.")
    
    return signal_copy


# ==============================================
#           spectrogram augmentations
# ==============================================
def TimeStretch(signal: torch.tensor, n_fft: int = 201):
    '''
    Adapted from https://pytorch.org/audio/stable/generated/torchaudio.transforms.TimeStretch.html#torchaudio.transforms.TimeStretch
    NOTE: input spectrogram and return spectrogram
    TODO: need to check!!
    https://iver56.github.io/audiomentations/waveform_transforms/time_stretch/
    '''
    rate_max = 1.3
    rate_min = 0.7
    rate = np.random.uniform(rate_min, rate_max)

    stretch = torchaudio.transforms.TimeStretch(n_freq = n_fft)
    stretched_signal = stretch(signal, rate)

    return stretched_signal


def SpecMask(signal: torch.tensor):
    '''
    Apply masking to a spectrogram in the time and frequency domain
    Adapted from: https://pytorch.org/audio/stable/generated/torchaudio.transforms.TimeMasking.html#torchaudio.transforms.TimeMasking
    NOTE: input spectrogram and return spectrogram
    '''
    device = signal.device
    dtype = signal.dtype
    signal_copy = signal.clone()

    fill_value = 0.
    shape = signal.size()
    length = shape[-1]
    wide = shape[-2]

    # time domain
    signal_copy = signal_copy.reshape([-1] + list(shape[-2:]))
    mask_factor_time = np.random.uniform(0.0, 0.2)
    mask_length_time = mask_factor_time * length
    mask_length_time = mask_length_time if mask_length_time else 1
    start_point = int(torch.rand(1)*(length - mask_length_time))
    end_point = start_point + mask_length_time

    mask_time = torch.arange(0, signal_copy.shape[2], device=device)
    mask_time = (mask_time >= start_point) & (mask_time < end_point)
    
    signal_copy = signal_copy.masked_fill(mask_time, fill_value)

    # frequency mask
    mask_factor_freq = np.random.uniform(0.0, 0.2)
    energy = signal.sum(dim=-1) / length
    probability = exponential(energy)
    _, prob_indices = torch.sort(input=probability, descending=True, dim=-1)
    threshold_idx = int(wide * mask_factor_freq)
    threshold = energy[:,prob_indices[:,threshold_idx]]
    mask_freq = energy < threshold
    mask_freq = mask_freq.unsqueeze(-1)
    mask_freq.to(device)
    signal_copy = signal_copy.masked_fill(mask_freq, fill_value)

    signal_copy = signal_copy.reshape(shape[:-2] + signal_copy.shape[-2:])

    return signal_copy

def SpecFlip(signal: torch.tensor):
    '''
    Horizontally flip the spectrogram.
    NOTE: input spectrogram
    '''
    return signal.flip(dims=[2])


def tf_mask(signal: torch.tensor, sigma: float):
    mask_id = torch.from_numpy(np.random.binomial(1, sigma, size=signal.size())).to(torch.bool)
    signal_copy = signal.clone()
    signal_copy[~mask_id] = 0
    return signal_copy

# ==============================================
#                   Compose
# ==============================================
def random_waveform_transform(signal: torch.tensor):
    if probability() < 0.8:
        signal = AddGaussianSNR(signal, 0, 10)
    
    # if probability() > 0.5:
    #    signal = Shift(signal)
    
    # if probability() > 0.7:
    #     signal = TimeMask(signal)
    
    # if probability() > 0.5:
    # signal = Gain(signal)

    # if probability() > 0.5:
    #    signal = Flip(signal)
    
    return signal

def random_spectrogram_transform(signal: torch.tensor, n_fft: int=150):
    '''
    NOTE: input waveform signal
    '''
    signal = STFT(signal, n_fft=n_fft)

    # if probability() < 0.6:
    #     signal = tf_mask(signal, 0.6)

    # if probability() > 0.5:
        # signal = SpecMask(signal)

    # if probability() < 0.5:
    #     signal = SpecFlip(signal)

    return signal

# ==============================================
#              Compared methods
# ==============================================
def frequency_transforms(signal: torch.tensor):
    '''
    NOTE: input waveform signal.
    '''
    signal = FFT(signal)
    
    return signal