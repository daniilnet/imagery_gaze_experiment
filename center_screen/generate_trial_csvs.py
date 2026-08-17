"""
generate_trial_csvs.py

Generates the fixed pseudorandom trial sequences used by
center_screen_imagery.py (imagery task) and center_screen_perception.py
(perception task):

    training_trials.csv              (4 trials  - imagery training)
    imagery_trials.csv               (60 trials - imagery task)
    perception_training_trials.csv   (4 trials  - perception training)
    perception_trials.csv            (10 trials - perception task)

Stimuli are shown in the center of the screen -- there is no left/right
placement logic at all; only the presentation (preview) order and the
cuing order are counterbalanced:
    - cue / cued_image            : H or F, 50/50 split, no more than
                                     MAX_RUN identical cues in a row.
    - first_image / second_image  : the order of the two-stimulus preview
                                     shown at the start of every trial
                                     (face_center.png then house_center.png,
                                     or vice-versa), also 50/50 with the
                                     same run constraint.

Deterministic: re-running this script regenerates identical CSVs (fixed
per-phase seeds), so every participant sees the same sequences.
"""
import csv
import os
import random

MAX_RUN = 2  # longest allowed run of identical consecutive values

CUE_TO_IMAGE = {
    "H": "house_center.png",
    "F": "face_center.png",
}

ORDER_TO_IMAGES = {
    "FH": ("face_center.png", "house_center.png"),
    "HF": ("house_center.png", "face_center.png"),
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


def _shuffle_balanced_sequence(n, seed, labels, max_attempts=100000):
    """Original shuffle-and-reject generator, kept as the primary path so
    already-generated CSVs (already run with real participants) regenerate
    byte-for-byte identical output."""
    half = n // 2
    values = [labels[0]] * half + [labels[1]] * half
    rng = random.Random(seed)
    for _ in range(max_attempts):
        rng.shuffle(values)
        if longest_run(values) <= MAX_RUN:
            return values[:]
    return None


def _construct_balanced_sequence(n, seed, labels, max_attempts=1000):
    """Fallback generator: builds the sequence directly from randomized
    runs of length 1-2 (interleaved between labels) instead of shuffling
    and rejecting. For a strict 50/50 split, the odds of a random
    full-sequence shuffle satisfying MAX_RUN=2 collapse combinatorially as
    n grows (already near-zero by n=60), so shuffle-and-reject stops
    finding a match within any reasonable attempt budget."""
    half = n // 2
    rng = random.Random(seed)

    def make_runs(total):
        runs = []
        remaining = total
        while remaining > 0:
            size = rng.choice((1, 2)) if remaining > 1 else 1
            runs.append(size)
            remaining -= size
        return runs

    for _ in range(max_attempts):
        runs_a = make_runs(half)
        runs_b = make_runs(half)
        if abs(len(runs_a) - len(runs_b)) > 1:
            continue
        rng.shuffle(runs_a)
        rng.shuffle(runs_b)
        if rng.random() < 0.5:
            first, first_label, second, second_label = runs_a, labels[0], runs_b, labels[1]
        else:
            first, first_label, second, second_label = runs_b, labels[1], runs_a, labels[0]

        values = []
        i = j = 0
        while i < len(first) or j < len(second):
            if i < len(first):
                values.extend([first_label] * first[i])
                i += 1
            if j < len(second):
                values.extend([second_label] * second[j])
                j += 1

        if len(values) == n and longest_run(values) <= MAX_RUN:
            return values
    raise RuntimeError(f"Could not satisfy max-run constraint for n={n}, seed={seed}")


def generate_balanced_sequence(n, seed, labels):
    """n values split evenly between the two labels, such that no label
    repeats more than MAX_RUN times in a row."""
    if n % 2 != 0:
        raise ValueError(f"n must be even for an exact 50/50 split, got {n}")
    result = _shuffle_balanced_sequence(n, seed, labels)
    if result is not None:
        return result
    return _construct_balanced_sequence(n, seed, labels)


def write_csv(filename, n, cue_seed, order_seed):
    cues   = generate_balanced_sequence(n, cue_seed,   ("H", "F"))
    orders = generate_balanced_sequence(n, order_seed, ("FH", "HF"))

    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_num", "cue", "cued_image", "first_image", "second_image"])
        for i, (cue, order) in enumerate(zip(cues, orders), start=1):
            first_image, second_image = ORDER_TO_IMAGES[order]
            writer.writerow([i, cue, CUE_TO_IMAGE[cue], first_image, second_image])
    print(f"Saved: {path}  ({n} trials, cue run <= {longest_run(cues)}, "
          f"order run <= {longest_run(orders)})")


if __name__ == "__main__":
    write_csv("training_trials.csv",           n=4,  cue_seed=42, order_seed=142)
    write_csv("imagery_trials.csv",             n=60, cue_seed=43, order_seed=143)
    write_csv("perception_training_trials.csv", n=4,  cue_seed=44, order_seed=144)
    write_csv("perception_trials.csv",          n=10, cue_seed=45, order_seed=145)
