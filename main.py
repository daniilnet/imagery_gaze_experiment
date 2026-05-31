"""
main.py

Dependencies:
    requires python 3.10.11 specifically!
    downloaded from https://www.python.org/downloads/windows/

    pip install psychopy lxml pygame
    pip install https://github.com/esdalmaijer/PyGaze/archive/refs/heads/master.zip
    (notice there are different 'pygaze' packages, download from the address above)

Folder structure expected:
    pool/
        face_left.png       face_right.png
        house_left.png      house_right.png
        3blank.jpg          4mask.jpg

Eye tracker: Gazepoint (OpenGaze protocol).

Design:
    Training:   4 trials (one per condition, no ET), before main experiment.

    Imagery:    4 conditions x 16 trials = 64 trials, split into two halves.
                Each condition: one face + one house image.
                Retrocue counterbalanced: 8 cue=1, 8 cue=2 per condition.

    Perception: 4 conditions x 4 trials = 16 trials.
                Same as imagery but step 7 shows the cued image instead of blank.

Trial sequence (timings):
    1. Start fixation cross          1000 ms
    2. Image 1                       1300 ms
    3. Blank (no cross)               500 ms
    4. Image 2                       1300 ms
    5. Blank after image 2            500 ms
    6. Retrocue number                140 ms
    7. Blank gaze / cued image       2000 ms   (imagery / perception)
    8. Vividness rating (1-4)        until keypress
    9. ITI blank                     1000 ms
"""

import os
import random
import csv
from datetime import datetime

# -- Set BEFORE any pygaze imports --------------------------------------------
os.environ["DISPTYPE"] = "psychopy"
os.environ["TRACKERTYPE"] = "opengaze"

# -- PsychoPy -----------------------------------------------------------------
from psychopy import visual, core, event

# -- PyGaze -------------------------------------------------------------------
import pygaze
import pygaze.settings as pygaze_settings
from pygaze.display import Display
from pygaze.eyetracker import EyeTracker
from pygaze import libtime

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
SCREEN_W    = 1920
SCREEN_H    = 1080   # change on lab computer
FULLSCREEN  = True
BACKGROUND  = "#7f7f7f"
FG_COLOR    = "white"
POOL_DIR = os.path.join(os.path.dirname(__file__), "pool")

# Timing (ms)
T_START_FIX     = 1500   # fixation cross before first image
T_IMG           = 2000   # each image on screen
T_BETWEEN_BLANK = 500    # blank between images (no cross)
T_POST_IMG2     = 500    # blank after image 2, replaces mask
T_RETROCUE      = 300    # retrocue number
T_BLANK_GAZE    = 3500   # blank gaze period
T_ITI           = 1500   # inter-trial interval


# -- Experiment design --------------------------------------------------------
# Each entry: (img_first, img_second, condition_label)
CONDITIONS = [
    ("face_right.png",  "house_left.png",  "face_R_house_L_face1st"),
    ("face_left.png",   "house_right.png", "face_L_house_R_face1st"),
    ("house_right.png", "face_left.png",   "house_R_face_L_house1st"),
    ("house_left.png",  "face_right.png",  "house_L_face_R_house1st"),
]
TRIALS_PER_CONDITION      = 16   # 8 cue=1 + 8 cue=2
HALF                      = 32   # trials in each imagery half
N_TRAINING                = 4    # one per condition
PERCEPTION_PER_CONDITION  = 4    # 16 perception trials total (2 cue=1 + 2 cue=2)

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def pool(filename):
    return os.path.join(POOL_DIR, filename)


def wait_ms(ms):
    core.wait(ms / 1000.0, hogCPUperiod=ms / 1000.0)


def draw_cross(win):
    v = visual.Line(win, start=(0, -16), end=(0, 16), color=FG_COLOR, colorSpace='rgb', lineWidth=1)
    h = visual.Line(win, start=(-16, 0), end=(16, 0), color=FG_COLOR, colorSpace='rgb', lineWidth=1)
    v.draw(); h.draw()
    return win.flip()


def draw_blank(win):
    return win.flip()   # background color already set on the window


def draw_text(win, text, height=30):
    stim = visual.TextStim(win, text=text, color=FG_COLOR, height=height,
                           wrapWidth=1400, alignText='center', anchorHoriz='center')
    stim.draw()
    return win.flip()


def wait_keypress(win, keys=None):
    """Wait for one of `keys`; quit on Escape."""
    event.clearEvents()
    while True:
        pressed = event.getKeys(keyList=(keys or []) + ['escape'])
        if pressed:
            if pressed[0] == 'escape':
                core.quit()
            return pressed[0]


def draw_image(win, filename, size=None):
    img = visual.ImageStim(win, image=pool(filename), size=size)
    img.draw()
    return win.flip()


def save_csv(log_rows, log_file):
    if not log_rows:
        return
    with open(log_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Log saved -> {log_file}")


# -----------------------------------------------------------------------------
# VIVIDNESS RATING
# -----------------------------------------------------------------------------
def get_vividness(win):
    """Display vividness scale and wait for key 1-4. Returns int rating."""
    draw_text(win,
              "How vivid was your mental image?\n\n"
              "1 = No image at all\n"
              "2 = Vague / dim\n"
              "3 = Moderately clear\n"
              "4 = Perfectly clear and vivid",
              height=28)
    key = wait_keypress(win, keys=['1', '2', '3', '4'])
    return int(key)



# -----------------------------------------------------------------------------
# TRIAL LIST GENERATION  (counterbalanced)
# -----------------------------------------------------------------------------
def build_trial_list():
    """
    80 trials: each condition gets exactly 10 cue=1 + 10 cue=2, then
    the full list is shuffled.
    """
    trials = []
    for img1, img2, label in CONDITIONS:
        cues = [1] * (TRIALS_PER_CONDITION // 2) + [2] * (TRIALS_PER_CONDITION // 2)
        random.shuffle(cues)
        for cue in cues:
            trials.append({
                "img_first":       img1,
                "img_second":      img2,
                "condition_label": label,
                "img_for_gaze":    cue,
            })
    random.shuffle(trials)
    return trials


def build_training_list():
    """4 trials, one per condition, random cue, shuffled."""
    trials = []
    for img1, img2, label in CONDITIONS:
        trials.append({
            "img_first":       img1,
            "img_second":      img2,
            "condition_label": label,
            "img_for_gaze":    random.choice([1, 2]),
        })
    random.shuffle(trials)
    return trials


def build_perception_list():
    """16 trials: each condition gets exactly 2 cue=1 + 2 cue=2, shuffled."""
    trials = []
    for img1, img2, label in CONDITIONS:
        cues = [1, 1, 2, 2]
        random.shuffle(cues)
        for cue in cues:
            trials.append({
                "img_first":       img1,
                "img_second":      img2,
                "condition_label": label,
                "img_for_gaze":    cue,
            })
    random.shuffle(trials)
    return trials


# -----------------------------------------------------------------------------
# CORE TRIAL SEQUENCE
# -----------------------------------------------------------------------------
def run_trial_sequence(win, tracker, trial_num, trial_def,
                       log_rows, is_training=False, mode='imagery'):
    """
    Runs one full trial.
    tracker=None  -> skips all ET calls (training).
    mode='imagery'    -> step 7 is a blank gaze period.
    mode='perception' -> step 7 shows the cued image.
    Returns the vividness rating (int 1-5).
    """
    img1            = trial_def["img_first"]
    img2            = trial_def["img_second"]
    img_for_gaze    = trial_def["img_for_gaze"]
    condition_label = trial_def["condition_label"]
    cued_image      = img1 if img_for_gaze == 1 else img2
    tag             = "training" if is_training else f"{mode}_{trial_num}"

    # -- 1. Start fixation cross (1000 ms) ------------------------------------
    if tracker:
        tracker.start_recording()
    t0_fix = draw_cross(win)
    if tracker:
        tracker.log(f"{tag}_StartFixation_at_{t0_fix}")
    wait_ms(T_START_FIX)
    if tracker:
        tracker.log(f"{tag}_EndFixation_at_{libtime.get_time()}")

    # -- 2. Image 1 (2000 ms) -------------------------------------------------
    t0 = draw_image(win, img1)
    if tracker:
        tracker.log(f"{tag}_StartImage1_{img1}_at_{t0}")
    wait_ms(T_IMG)
    if tracker:
        tracker.log(f"{tag}_EndImage1_at_{libtime.get_time()}")

    # -- 3. Blank between images (500 ms, no cross) ---------------------------
    draw_blank(win)
    wait_ms(T_BETWEEN_BLANK)

    # -- 4. Image 2 (2000 ms) -------------------------------------------------
    t0 = draw_image(win, img2)
    if tracker:
        tracker.log(f"{tag}_StartImage2_{img2}_at_{t0}")
    wait_ms(T_IMG)
    if tracker:
        tracker.log(f"{tag}_EndImage2_at_{libtime.get_time()}")

    # -- 5. Blank after image 2 (500 ms) --------------------------------------
    draw_blank(win)
    wait_ms(T_POST_IMG2)

    # -- 6. Retrocue number (140 ms) ------------------------------------------
    t0 = draw_text(win, str(img_for_gaze), height=24)
    if tracker:
        tracker.log(f"{tag}_StartRetroCue_{img_for_gaze}_at_{t0}")
    wait_ms(T_RETROCUE)
    if tracker:
        tracker.log(f"{tag}_EndRetroCue_at_{libtime.get_time()}")

    # -- 7. Blank gaze (imagery) or cued image (perception) for 2000 ms -------
    if mode == 'perception':
        t0 = draw_image(win, cued_image)
        if tracker:
            tracker.log(f"{tag}_StartPerceptionImage_{cued_image}_at_{t0}")
    else:
        t0 = draw_blank(win)
        if tracker:
            tracker.log(f"{tag}_StartBlankGaze_cued_{img_for_gaze}_at_{t0}")
    wait_ms(T_BLANK_GAZE)
    if tracker:
        tracker.log(f"{tag}_EndStep7_at_{libtime.get_time()}")

    # -- 8. Vividness rating (imagery only) --------------------------------------
    if mode == 'perception':
        vividness = None
    else:
        if tracker:
            tracker.log(f"{tag}_StartVividnessRating")
        vividness = get_vividness(win)
        if tracker:
            tracker.log(f"{tag}_VividnessRating_{vividness}")

    # -- 9. ITI blank (1000 ms) -----------------------------------------------
    draw_blank(win)
    wait_ms(T_ITI)

    # -- ET: stop recording & log variables -----------------------------------
    if tracker:
        tracker.stop_recording()
        tracker.log_var("phase",                 mode)
        tracker.log_var("trial_num",             trial_num)
        tracker.log_var("condition_label",       condition_label)
        tracker.log_var("img_first",             img1)
        tracker.log_var("img_second",            img2)
        tracker.log_var("ImgForGaze_1st_or_2nd", img_for_gaze)
        tracker.log_var("cued_image",            cued_image)
        if vividness is not None:
            tracker.log_var("vividness",         vividness)

    # -- CSV log (non-training trials only) -----------------------------------
    if not is_training:
        row = {
            "phase":                 mode,
            "trial_num":             trial_num,
            "condition_label":       condition_label,
            "img_first":             img1,
            "img_second":            img2,
            "ImgForGaze_1st_or_2nd": img_for_gaze,
            "cued_image":            cued_image,
            "vividness":             vividness,
        }
        log_rows.append(row)

    return vividness


# -----------------------------------------------------------------------------
# TRAINING SESSION
# -----------------------------------------------------------------------------
def run_training(win):
    """
    4 practice trials, no eye tracker. Robust transition to main experiment:
    - events are cleared before and after
    - window is left blank and stable before returning
    """
    draw_text(win,
              "PRACTICE\n\n"
              "You will now see a few practice trials.\n\n"
              "Press SPACE to begin.")
    wait_keypress(win, keys=['space'])

    for i, trial_def in enumerate(build_training_list(), start=1):
        run_trial_sequence(win, tracker=None, trial_num=i,
                           trial_def=trial_def, log_rows=[],
                           is_training=True)

    # -- Clean transition: blank screen, flush events, then show message ------
    draw_blank(win)
    event.clearEvents()
    core.wait(0.5)   # brief pause so last ITI doesn't bleed into next screen

    draw_text(win,
              "Practice complete!\n\n"
              "The main experiment will now begin.\n\n"
              "Press SPACE to continue.")
    wait_keypress(win, keys=['space'])

    draw_blank(win)
    event.clearEvents()


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    # -- Subject number prompt (terminal, before any window opens) ------------
    while True:
        try:
            subject_nr = int(input("Enter subject number: "))
            break
        except ValueError:
            print("Please enter a valid integer.")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file  = f"log_subj_{subject_nr}_{timestamp}.csv"
    et_log    = f"gazepoint_data_subj_{subject_nr}_{timestamp}"

    # -- Let PyGaze own the window (prevents double-window on startup) --------
    pygaze_settings.DISPSIZE   = (SCREEN_W, SCREEN_H)
    pygaze_settings.SCREENNR   = 0
    pygaze_settings.FULLSCREEN = FULLSCREEN
    pygaze_settings.BGCOLOUR   = (127, 127, 127)  # standard gray

    disp    = Display()
    win     = pygaze.expdisplay  # window PyGaze created, stored on the module
    tracker = EyeTracker(disp, trackertype="opengaze", logfile=et_log)

    # -- Training -------------------------------------------------------------
    run_training(win)

    # -- Post-training: continue or quit --------------------------------------
    draw_text(win,
              "Practice complete!\n\n"
              "Press  C  to start the main experiment\n"
              "Press  Q  to quit.")
    key = wait_keypress(win, keys=['c', 'q'])
    if key == 'q':
        tracker.close()
        disp.close()
        win.close()
        core.quit()

    # -- Build trial lists ----------------------------------------------------
    all_trials       = build_trial_list()
    perception_trials = build_perception_list()
    log_rows   = []
    trial_num  = 1

    # =========================================================================
    # FIRST HALF  (trials 1-40)
    # =========================================================================
    draw_text(win, "Part 1 of 2\n\nPress SPACE to begin.")
    wait_keypress(win, keys=['space'])

    for trial_def in all_trials[:HALF]:
        run_trial_sequence(win, tracker, trial_num, trial_def, log_rows)
        trial_num += 1

    # -- Mid-point save -------------------------------------------------------
    save_csv(log_rows, log_file)

    # -- Mid-point prompt: continue or end ------------------------------------
    draw_text(win,
              f"Part 1 complete  ({HALF} of {HALF * 2} trials).\n"
              "Your results have been saved.\n\n"
              "Press  C  to continue to Part 2\n"
              "Press  Q  to end the session now.")
    key = wait_keypress(win, keys=['c', 'q'])

    if key == 'q':
        draw_text(win, "Session ended. Thank you!\n\nPress SPACE to exit.")
        wait_keypress(win, keys=['space'])
        tracker.close()
        disp.close()
        win.close()
        core.quit()

    # =========================================================================
    # SECOND HALF  (trials 41-80)
    # =========================================================================
    draw_text(win, "Part 2 of 2\n\nPress SPACE to begin.")
    wait_keypress(win, keys=['space'])

    for trial_def in all_trials[HALF:]:
        run_trial_sequence(win, tracker, trial_num, trial_def, log_rows)
        trial_num += 1

    # -- Save after imagery ---------------------------------------------------
    save_csv(log_rows, log_file)

    # =========================================================================
    # PERCEPTION  (40 trials)
    # =========================================================================
    draw_text(win,
              "Perception section\n\n"
              "The procedure is the same, but after the cue\n"
              "the corresponding image will be shown on screen.\n\n"
              "Press SPACE to begin.")
    wait_keypress(win, keys=['space'])

    for trial_def in perception_trials:
        run_trial_sequence(win, tracker, trial_num, trial_def,
                           log_rows, mode='perception')
        trial_num += 1

    # -- Final save (imagery + perception) ------------------------------------
    save_csv(log_rows, log_file)

    # -- End screen -----------------------------------------------------------
    draw_text(win, "Experiment complete!  Thank you.\n\nPress SPACE to exit.")
    wait_keypress(win, keys=['space'])

    # -- Cleanup --------------------------------------------------------------
    tracker.close()
    disp.close()
    win.close()
    core.quit()


if __name__ == "__main__":
    main()
