"""
one_sided_imagery.py -- Imagery task

Run this as its own session. The perception task lives entirely in
one_sided_perception.py; the two scripts do not depend on each other.
Shared setup, timing, and trial-running logic live in
one_sided_experiment_common.py.

Design:
    Every trial opens with a two-stimulus preview (face_right.png and
    house_right.png, back to back) before the imagery cue -- there is no
    separate block-intro step. The preview order is counterbalanced across
    trials (see generate_trial_csvs.py).

    Training:  4 practice trials (no ET, not logged).
    Imagery:   60 trials. Break screen after every 10 trials (except the
               last).

See one_sided_experiment_common.py for the full per-trial timing breakdown.
"""

from one_sided_experiment_common import (
    draw_text,
    load_trials,
    make_log_paths,
    prompt_subject_number,
    run_training,
    run_trials,
    save_csv,
    setup_display_and_tracker,
    wait_keypress,
    wait_start_keypress,
)
from psychopy import core

# -----------------------------------------------------------------------------
# DEV MODE SWITCH
# -----------------------------------------------------------------------------
without_tracker = 1   # 1 = run experiment without eye tracker connected (testing).
                      # On-screen is identical to normal runtime; no ET/gaze data is logged.

BREAK_EVERY_TRIALS = 10   # rest break after every N imagery trials

# -----------------------------------------------------------------------------
# FIXED PSEUDORANDOM TRIAL ORDERS (see generate_trial_csvs.py)
# Same cue / preview-order sequence for every participant; counterbalanced
# 50/50 with no more than 2 identical values in a row.
# -----------------------------------------------------------------------------
FIXED_TRAINING_TRIALS = load_trials("training_trials.csv")
FIXED_IMAGERY_TRIALS  = load_trials("imagery_trials.csv")


def main():
    subject_nr = prompt_subject_number(without_tracker)
    log_file, et_log = make_log_paths("imagery", subject_nr)

    disp, win, tracker = setup_display_and_tracker(without_tracker, et_log)

    # -- Training -------------------------------------------------------------
    draw_text(win, "Training\n\nPress SPACE to begin.")
    wait_keypress(win, keys=['space'])
    run_training(win, FIXED_TRAINING_TRIALS)

    # -- Experiment start -----------------------------------------------------
    log_rows = []
    draw_text(win, "Experiment\n\nPress SPACE to begin.")
    wait_start_keypress(win, tracker, disp, log_rows, log_file, without_tracker)

    # -- Imagery trials, with a break every BREAK_EVERY_TRIALS trials ----------
    run_trials(win, tracker, FIXED_IMAGERY_TRIALS, log_rows,
               start_trial_num=1, mode='imagery',
               break_every=BREAK_EVERY_TRIALS,
               disp=disp, log_file=log_file, without_tracker=without_tracker)

    # -- Final save -------------------------------------------------------------
    save_csv(log_rows, log_file, without_tracker)

    # -- End screen -----------------------------------------------------------
    draw_text(win, "Experiment complete!  Thank you.\n\nPress SPACE to exit.")
    wait_keypress(win, keys=['space'])

    # -- Cleanup --------------------------------------------------------------
    if tracker:
        tracker.close()
    disp.close()
    win.close()
    core.quit()


if __name__ == "__main__":
    main()
