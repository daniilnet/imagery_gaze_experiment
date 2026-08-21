"""
center_screen_perception.py -- Perception task

Run this as its own session, entirely separate from center_screen_imagery.py
(imagery task). Shared setup, timing, and trial-running logic live in
center_screen_experiment_common.py.

Design:
    Stimuli are shown in the center of the screen -- no left/right logic at
    all. Every trial opens with the same two-stimulus preview used in the
    imagery task (face_center.png and house_center.png, back to back), in a
    counterbalanced order (see generate_trial_csvs.py). The cued image is
    then shown during the "imagery" period instead of a blank screen; no
    vividness / time-to-imagine ratings are collected.

    Perception: 10 trials, run in a single block (no breaks).

See center_screen_experiment_common.py for the full per-trial timing breakdown.
"""

from center_screen_experiment_common import (
    draw_text,
    load_trials,
    make_log_paths,
    prompt_subject_number,
    quit_and_save,
    run_trials,
    save_csv,
    setup_display_and_tracker,
    wait_keypress,
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
FIXED_PERCEPTION_TRIALS = load_trials("perception_trials.csv")


def main():
    subject_nr = prompt_subject_number(without_tracker)
    log_file, et_log = make_log_paths("perception", subject_nr)

    disp, win, tracker = setup_display_and_tracker(without_tracker, et_log)

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

    # Cleanup -- quit_and_save() closes the display exactly once (disp and win
    # are the same PsychoPy window) and always closes the tracker, which is
    # what lets OpenGaze's non-daemon threads exit.
    quit_and_save(win, tracker, disp, log_rows, log_file, without_tracker)


if __name__ == "__main__":
    main()
