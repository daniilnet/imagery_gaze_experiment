"""
generate_trial_csvs.py

Generates the fixed pseudorandom H/F cue sequences used by main.py:
    training_trials.csv    (1 block  x 6 trials)
    imagery_trials.csv     (10 blocks x 6 trials = 60 trials)
    perception_trials.csv  (2 blocks x 6 trials = 12 trials)

Each block is counterbalanced (3x H, 3x F) and shuffled such that no
cue repeats more than 2 times in a row, checked across block boundaries
too (using the trailing cues of the previous block).

Deterministic: re-running this script regenerates identical CSVs (fixed
per-phase seeds), so every participant sees the same sequences.
"""
import csv
import os
import random

MAX_RUN = 2  # longest allowed run of identical consecutive cues

CUE_TO_IMAGE = {
    "H": "house_right.png",
    "F": "face_right.png",
}


def longest_run(seq):
    if not seq:
        return 0
    longest = current = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def generate_sequence(n_blocks, seed, max_attempts_per_block=1000):
    rng = random.Random(seed)
    sequence = []
    for _ in range(n_blocks):
        for _attempt in range(max_attempts_per_block):
            block = ["H", "H", "H", "F", "F", "F"]
            rng.shuffle(block)
            tail = sequence[-MAX_RUN:] + block
            if longest_run(tail) <= MAX_RUN:
                sequence.extend(block)
                break
        else:
            raise RuntimeError(f"Could not satisfy max-run constraint for block starting at trial {len(sequence) + 1}")
    return sequence


def write_csv(filename, n_blocks, seed):
    sequence = generate_sequence(n_blocks, seed)
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["block", "trial_in_block", "cue", "cued_image"])
        for i, cue in enumerate(sequence):
            block = i // 6 + 1
            trial_in_block = i % 6 + 1
            writer.writerow([block, trial_in_block, cue, CUE_TO_IMAGE[cue]])
    print(f"Saved: {path}  ({n_blocks} blocks, {len(sequence)} trials, longest run = {longest_run(sequence)})")


if __name__ == "__main__":
    write_csv("training_trials.csv",   n_blocks=1,  seed=42)
    write_csv("imagery_trials.csv",    n_blocks=10, seed=43)
    write_csv("perception_trials.csv", n_blocks=2,  seed=44)
