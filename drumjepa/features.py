"""State and action features for drum-JEPA.

State: log-mel spectrogram following Music-JEPA and ZZWaang/audio2midi:
16 kHz, n_fft 2048, hop 160 (10 ms frames), 229 HTK mel bins from 30 Hz to
8 kHz, no filterbank norm, power spectrogram, log(clamp(mel, 1e-5)).
A 2 s segment is 200 frames x 229 bins.

Action: drumroll (frames x K_V1) with velocity/127 in the onset cell (max if
two hits share a cell), plus the hi-hat pedal position CC4/127 forward-filled
per frame (back-filled from the first CC4 message before it). Both come from
the CANONICAL MIDI of the sequence (see docs/inventory.md).
"""
import librosa
import mido
import numpy as np

from drumjepa.drum_map import CLASS_INDEX_V1, DRUM_MAP_V1, HH_POSITION_CC, K_V1

SR = 16000
N_FFT = 2048
HOP = 160
N_MELS = 229
F_MIN = 30
F_MAX = 8000
LOG_FLOOR = 1e-5
FRAMES_PER_S = SR // HOP  # 100
SEG_FRAMES = 200          # 2 s

MEL_PARAMS = dict(sr=SR, n_fft=N_FFT, hop=HOP, n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX,
                  mel_scale="htk", norm=None, power=2, log="log(clamp(x, 1e-5))", center=True)


def log_mel(y: np.ndarray, sr: int) -> np.ndarray:
    """Waveform (any sr, mono float) -> (n_frames, N_MELS) float32 log-mel."""
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
                                       fmin=F_MIN, fmax=F_MAX, htk=True, norm=None, power=2.0,
                                       center=True)
    return np.log(np.maximum(m, LOG_FLOOR)).T.astype(np.float32)


def midi_to_action(path: str, n_frames: int):
    """Canonical MIDI -> (roll (n_frames, K_V1), cc4 (n_frames,)) float32.

    Frames past the MIDI end are zeros for the roll and hold the last CC4 value.
    Raises KeyError on a pitch missing from DRUM_MAP_V1 (only per-kit remapped
    files contain such pitches; never pass those).
    """
    roll = np.zeros((n_frames, K_V1), np.float32)
    cc_t, cc_v = [], []
    t = 0.0
    for msg in mido.MidiFile(path):
        t += msg.time
        f = int(t * FRAMES_PER_S)
        if msg.type == "note_on" and msg.velocity > 0:
            k = CLASS_INDEX_V1[DRUM_MAP_V1[msg.note]]  # KeyError on unmapped pitch
            if f < n_frames:
                roll[f, k] = max(roll[f, k], msg.velocity / 127.0)
        elif msg.type == "control_change" and msg.control == HH_POSITION_CC:
            cc_t.append(f)
            cc_v.append(msg.value / 127.0)
    cc4 = np.zeros(n_frames, np.float32)
    if cc_t:
        cc_t = np.minimum(np.array(cc_t), n_frames - 1)
        cc4[:] = cc_v[0]                      # back-fill before the first message
        cc4[cc_t] = cc_v                      # later messages in the same frame win
        change = np.zeros(n_frames, bool)
        change[cc_t] = True
        idx = np.where(change, np.arange(n_frames), 0)
        np.maximum.accumulate(idx, out=idx)  # forward-fill
        cc4 = cc4[idx]
    return roll, cc4
