import librosa
import numpy as np
import pretty_midi
import mir_eval.melody


def get_highest_pitches_from_piano_roll(pr, velocity_threshold=10):
    """
    params:
        pr : (128, time(frame)) - contains velocity values
        velocity_threshold : int - minimum velocity to consider as a valid note

    return:
        highest_pitches : (time(frame), )
    """
    highest_pitches = []
    for i in range(pr.shape[1]):
        # Filter notes with velocity below threshold (assumed to be noise/ghost notes)
        notes = pr[:, i]
        valid_indices = np.where(notes >= velocity_threshold)[0]

        if len(valid_indices) == 0:
            highest_pitches.append(np.nan)
        else:
            # Skyline approach: Pick the highest pitch among valid notes
            highest_pitches.append(valid_indices[-1])
    highest_pitches = np.array(highest_pitches)

    return highest_pitches


def _f0(y, sr, hop_length):
    pyin = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
        hop_length=hop_length,
    )
    # f0, voiced_flag, voiced_probs
    return pyin


def _evaluate_melody(midi, f0, sr, hop_length):
    x_coords = np.arange(0, midi.get_end_time(), hop_length / sr)
    pr = midi.get_piano_roll(fs=sr / hop_length, times=x_coords)
    highest_pitches = get_highest_pitches_from_piano_roll(pr)

    (ref_v, ref_c, est_v, est_c) = mir_eval.melody.to_cent_voicing(
        x_coords, f0, x_coords, librosa.midi_to_hz(highest_pitches)
    )

    raw_chroma = mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c)
    raw_pitch = mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c, est_v, est_c)
    return raw_chroma, raw_pitch


def evaluate_melody(
    midi: pretty_midi.PrettyMIDI,
    vocals: np.array,
    sr: int = 44100,
    hop_length: int = 1024,
):
    f0, voiced_flag, voiced_probs = _f0(vocals, sr=sr, hop_length=hop_length)

    raw_chroma, raw_pitch = _evaluate_melody(midi, f0, sr, hop_length)
    return raw_chroma, raw_pitch


if __name__ == "__main__":
    pass
