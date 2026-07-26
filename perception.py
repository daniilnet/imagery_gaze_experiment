"""
perception.py -- Perception task

Run this as its own session, entirely separate from main.py (imagery task).
Shared setup, timing, and trial-running logic live in experiment_common.py.

Design:
    Every trial opens with the same two-stimulus preview used in the imagery
    task (face_right.png and house_right.png, back to back), in a
    counterbalanced order (see generate_trial_csvs.py). The cued image is
    then shown during the "imagery" period instead of a blank screen; no
    vividness / time-to-imagine ratings are collected.

    Training:   4 practice trials (no ET, not logged), run in perception
                mode so their structure (cued image shown, no ratings)
                matches the perception trials.
    Perception: 10 trials, run in a single block (no breaks).

See experiment_common.py for the full per-trial timing breakdown.
"""

from psychopy import core

from experiment_common import (
    load_trials, draw_text, wait_keypress, save_csv,
    run_training, run_trials, setup_display_and_tracker,
    prompt_subject_number, make_log_paths,
)

# -----------------------------------------------------------------------------
# DEV MODE SWITCH
# -----------------------------------------------------------------------------
without_tracker = 1   # 1 = run experiment without eye tracker connected (testing).
                      # On-screen is identical to normal runtime; no ET/gaze data is logged.

# -----------------------------------------------------------------------------
# FIXED PSEUDORANDOM TRIAL ORDERS (see generate_trial_csvs.py)
# Same cue / preview-order sequence for every participant; counterbalanced
# 50/50 with no more than 2 identical values in a row.
# -----------------------------------------------------------------------------
FIXED_TRAINING_TRIALS   = load_trials("perception_training_trials.csv")
FIXED_PERCEPTION_TRIALS = load_trials("perception_trials.csv")


def main():
    subject_nr = prompt_subject_number(without_tracker)
    log_file, et_log = make_log_paths("perception", subject_nr)

    disp, win, tracker = setup_display_and_tracker(without_tracker, et_log)

    # -- Training -------------------------------------------------------------
    draw_text(win, "Training\n\nPress SPACE to begin.")
    wait_keypress(win, keys=['space'])
    run_training(win, FIXED_TRAINING_TRIALS, mode='perception')

    # -- Perception start -------------------------------------------------------
    draw_text(win, "Perception section\n\nPress SPACE to begin.")
    wait_keypress(win, keys=['space'])

    log_rows = []
    run_trials(win, tracker, FIXED_PERCEPTION_TRIALS, log_rows,
               start_trial_num=1, mode='perception',
               break_every=None,
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
