"""
generate_trial_csvs.py

Generates the fixed pseudorandom trial sequences used by main.py (imagery
task) and perception.py (perception task):

    training_trials.csv              (4 trials  - imagery training)
    imagery_trials.csv               (40 trials - imagery task)
    perception_training_trials.csv   (4 trials  - perception training)
    perception_trials.csv            (10 trials - perception task)

Each trial has two independently counterbalanced factors:
    - cue / cued_image            : H or F, 50/50 split, no more than
                                     MAX_RUN identical cues in a row.
    - first_image / second_image  : the order of the two-stimulus preview
                                     shown at the start of every trial
                                     (face_right.png then house_right.png,
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
    "H": "house_right.png",
    "F": "face_right.png",
}

ORDER_TO_IMAGES = {
    "FH": ("face_right.png", "house_right.png"),
    "HF": ("house_right.png", "face_right.png"),
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


def generate_balanced_sequence(n, seed, labels, max_attempts=100000):
    """n values split evenly between the two labels, shuffled such that no
    label repeats more than MAX_RUN times in a row."""
    if n % 2 != 0:
        raise ValueError(f"n must be even for an exact 50/50 split, got {n}")
    half = n // 2
    values = [labels[0]] * half + [labels[1]] * half
    rng = random.Random(seed)
    for _ in range(max_attempts):
        rng.shuffle(values)
        if longest_run(values) <= MAX_RUN:
            return values[:]
    raise RuntimeError(f"Could not satisfy max-run constraint for n={n}, seed={seed}")


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
    write_csv("imagery_trials.csv",             n=40, cue_seed=43, order_seed=143)
    write_csv("perception_training_trials.csv", n=4,  cue_seed=44, order_seed=144)
    write_csv("perception_trials.csv",          n=10, cue_seed=45, order_seed=145)
