"""
generate_trial_csvs.py

Generates the fixed pseudorandom trial sequences used by multi_stim_imagery.py
(imagery task) and multi_stim_perception.py (perception task):

    training_trials.csv              (8  trials - imagery training)
    imagery_trials.csv               (64 trials - imagery task)
    perception_training_trials.csv   (8  trials - perception training)
    perception_trials.csv            (16 trials - perception task)

Each trial independently counterbalances four factors:
    - face  (id, side)  : which of the 4 face identities is used, and
                           whether its "_left" or "_right" render is shown
                           -- 8 cells (4 ids x 2 sides), each cell used an
                           equal number of times, no cell repeated on
                           consecutive trials.
    - house (id, side)  : same as above, independently, for the 4 house
                           identities.
    - cue / cued_image   : H or F, 50/50 split, no more than MAX_RUN
                           identical cues in a row.
    - first_image/second_image : the order of the two-stimulus preview
                           shown at the start of every trial (face image
                           then house image, or vice-versa), also 50/50
                           with the same run constraint.

Trial counts are all multiples of 8 so the 8 face cells / 8 house cells
divide the session exactly evenly.

Deterministic: re-running this script regenerates identical CSVs (fixed
per-phase seeds), so every participant sees the same sequences.
"""
import csv
import os
import random

MAX_RUN = 2       # longest allowed run of identical consecutive cue/order values
FACE_IDS  = (1, 2, 3, 4)
HOUSE_IDS = (1, 2, 3, 4)
SIDES     = ("left", "right")

CUE_TO_CATEGORY = {"H": "house", "F": "face"}

ORDER_TO_CATEGORIES = {
    "FH": ("face", "house"),
    "HF": ("house", "face"),
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


def generate_balanced_sequence(n, seed, labels, max_run=MAX_RUN, max_attempts=200000):
    """n values split evenly across `labels` (any number of them), such
    that no label repeats more than `max_run` times in a row."""
    k = len(labels)
    if n % k != 0:
        raise ValueError(f"n={n} must be divisible by the number of labels ({k})")
    reps = n // k
    values = list(labels) * reps
    rng = random.Random(seed)
    for _ in range(max_attempts):
        rng.shuffle(values)
        if longest_run(values) <= max_run:
            return values[:]
    raise RuntimeError(f"Could not satisfy max-run constraint for n={n}, seed={seed}, labels={labels}")


def image_name(category, identity_id, side):
    return f"{category}{identity_id}_{side}.png"


def opposite_side(side):
    return "right" if side == "left" else "left"


def generate_ids_with_cell_constraint(n, seed, ids, side_seq, max_attempts=200000):
    """Balanced id sequence such that (id, side_seq[i]) never repeats the
    exact cell of the previous trial. Since side_seq is fixed in advance,
    that only requires a different id whenever the side repeats."""
    k = len(ids)
    if n % k != 0:
        raise ValueError(f"n={n} must be divisible by the number of ids ({k})")
    reps = n // k
    values = list(ids) * reps
    rng = random.Random(seed)
    for _ in range(max_attempts):
        rng.shuffle(values)
        if all(not (side_seq[i] == side_seq[i - 1] and values[i] == values[i - 1])
               for i in range(1, n)):
            return values[:]
    raise RuntimeError(f"Could not satisfy cell constraint for n={n}, seed={seed}, ids={ids}")


def write_csv(filename, n, cue_seed, order_seed, face_seed, house_seed):
    cues        = generate_balanced_sequence(n, cue_seed,   ("H", "F"))
    orders      = generate_balanced_sequence(n, order_seed, ("FH", "HF"))
    face_cells  = generate_balanced_sequence(
        n, face_seed,  [(fid, side) for fid in FACE_IDS for side in SIDES], max_run=1)
    # House always goes on the side opposite the face for that trial, so the
    # two stimuli never land on the same side. House id is then balanced
    # independently, with the same no-immediate-cell-repeat guarantee.
    house_sides = [opposite_side(side) for _, side in face_cells]
    house_ids   = generate_ids_with_cell_constraint(n, house_seed, HOUSE_IDS, house_sides)
    house_cells = list(zip(house_ids, house_sides))

    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial_num", "cue", "cued_image", "cued_side",
            "face_id", "face_side", "face_image",
            "house_id", "house_side", "house_image",
            "first_image", "second_image",
        ])
        for i, (cue, order, face_cell, house_cell) in enumerate(
                zip(cues, orders, face_cells, house_cells), start=1):
            face_id, face_side   = face_cell
            house_id, house_side = house_cell
            face_image  = image_name("face",  face_id,  face_side)
            house_image = image_name("house", house_id, house_side)

            cued_category = CUE_TO_CATEGORY[cue]
            cued_image = face_image if cued_category == "face" else house_image
            cued_side  = face_side  if cued_category == "face" else house_side

            first_category, second_category = ORDER_TO_CATEGORIES[order]
            images_by_category = {"face": face_image, "house": house_image}
            first_image  = images_by_category[first_category]
            second_image = images_by_category[second_category]

            writer.writerow([
                i, cue, cued_image, cued_side,
                face_id, face_side, face_image,
                house_id, house_side, house_image,
                first_image, second_image,
            ])
    print(f"Saved: {path}  ({n} trials, cue run <= {longest_run(cues)}, "
          f"order run <= {longest_run(orders)}, "
          f"face-cell run <= {longest_run(face_cells)}, "
          f"house-cell run <= {longest_run(house_cells)})")


if __name__ == "__main__":
    write_csv("training_trials.csv",           n=8,
              cue_seed=42, order_seed=142, face_seed=242, house_seed=342)
    write_csv("imagery_trials.csv",             n=64,
              cue_seed=43, order_seed=143, face_seed=243, house_seed=343)
    write_csv("perception_training_trials.csv", n=8,
              cue_seed=44, order_seed=144, face_seed=244, house_seed=344)
    write_csv("perception_trials.csv",          n=16,
              cue_seed=45, order_seed=145, face_seed=245, house_seed=345)
