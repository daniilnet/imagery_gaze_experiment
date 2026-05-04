"""
blank_img_gaze_v3.py

Dependencies:
    requires python 3.10.11 specifically!
    downloaded from https://www.python.org/downloads/windows/

    pip install psychopy PyGaze lxml pygame
    (notice that 'ppygaze' is a different repo. the correct one is 'PyGaze')

Folder structure expected:
    pool/
        face_left.png       face_right.png
        house_left.png      house_right.png
        3blank.jpg          4mask.jpg
        retrocue_1.png      retrocue_2.png

Eye tracker: Gazepoint (OpenGaze protocol).

Design (main experiment):
    4 conditions x 20 trials = 80 trials, split into two halves of 40.
    Each condition: one face + one house image, varying which category is
    shown first and which side each occupies.
    Retrocue (1 or 2) counterbalanced: 10 cue=1, 10 cue=2 per condition.

Trial sequence (timings):
    1. Drift correction
    2. Start fixation cross          1000 ms
    3. Image 1                       1300 ms
    4. Blank (no cross)               500 ms
    5. Image 2                       1300 ms
    6. Mask                           500 ms
    7. Retrocue image                 140 ms
    8. Blank gaze period             2000 ms
    9. Vividness rating (1-4)        until keypress
   10. ITI blank                     1000 ms
"""

import math
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
T_START_FIX     = 1000   # fixation cross before first image
T_IMG           = 1300   # each image on screen
T_BETWEEN_BLANK = 500    # blank between images (no cross)
T_MASK          = 500    # mask duration (no cross after)
T_RETROCUE      = 140    # retrocue image
T_BLANK_GAZE    = 2000   # blank gaze period
T_ITI           = 1000   # inter-trial interval

DRIFT_THRESHOLD_PX = 50

# -- Experiment design --------------------------------------------------------
# Each entry: (img_first, img_second, condition_label)
CONDITIONS = [
    ("face_right.png",  "house_left.png",  "face_R_house_L_face1st"),
    ("face_left.png",   "house_right.png", "face_L_house_R_face1st"),
    ("house_right.png", "face_left.png",   "house_R_face_L_house1st"),
    ("house_left.png",  "face_right.png",  "house_L_face_R_house1st"),
]
TRIALS_PER_CONDITION = 20   # 10 cue=1 + 10 cue=2
HALF              = 40      # trials in each session half

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
    return win.flip()   # background colour already set on the window


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
# DRIFT CORRECTION
# -----------------------------------------------------------------------------
def do_drift_correction(win, tracker):
    target_x, target_y = SCREEN_W / 2, SCREEN_H / 2
    while True:
        tracker.drift_correction(pos=(target_x, target_y), fix_triggered=True)
        gaze = tracker.sample()
        dist = math.sqrt((gaze[0] - target_x) ** 2 + (gaze[1] - target_y) ** 2)
        if dist <= DRIFT_THRESHOLD_PX:
            tracker.log(f"Drift_Check_Passed_Error_px_{dist:.2f}")
            return dist
        print(f"Drift dist {dist:.1f}px - look at centre")


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


# -----------------------------------------------------------------------------
# CORE TRIAL SEQUENCE
# -----------------------------------------------------------------------------
def run_trial_sequence(win, tracker, trial_num, trial_def,
                       log_rows, is_training=False):
    """
    Runs one full trial. tracker=None skips all ET calls (used in training).

    Returns the vividness rating (int 1-4).
    """
    img1            = trial_def["img_first"]
    img2            = trial_def["img_second"]
    img_for_gaze    = trial_def["img_for_gaze"]
    condition_label = trial_def["condition_label"]
    tag             = "training" if is_training else f"trial{trial_num}"

    # -- 1. Fixation cross + drift correction ------------------------------------
    draw_cross(win)                          # participant fixates before correction
    if tracker:
        do_drift_correction(win, tracker)
        tracker.start_recording()
    wait_ms(T_START_FIX)

    # -- 2. Image 1 (1300 ms) -------------------------------------------------
    t0 = draw_image(win, img1, size=(1251, 704))
    if tracker:
        tracker.log(f"{tag}_StartImage1_{img1}_at_{t0}")
    wait_ms(T_IMG)
    if tracker:
        tracker.log(f"{tag}_EndImage1_at_{libtime.get_time()}")

    # -- 3. Blank between images (500 ms, no cross) ---------------------------
    draw_blank(win)
    wait_ms(T_BETWEEN_BLANK)

    # -- 4. Image 2 (1300 ms) -------------------------------------------------
    t0 = draw_image(win, img2, size=(1251, 704))
    if tracker:
        tracker.log(f"{tag}_StartImage2_{img2}_at_{t0}")
    wait_ms(T_IMG)
    if tracker:
        tracker.log(f"{tag}_EndImage2_at_{libtime.get_time()}")

    # -- 5. Mask (500 ms, no cross after) -------------------------------------
    draw_image(win, "4mask.jpg", size=(850, 639))
    wait_ms(T_MASK)

    # -- 6. Retrocue number (140 ms) ------------------------------------------
    t0 = draw_text(win, str(img_for_gaze), height=24)
    if tracker:
        tracker.log(f"{tag}_StartRetroCue_{img_for_gaze}_at_{t0}")
    wait_ms(T_RETROCUE)
    if tracker:
        tracker.log(f"{tag}_EndRetroCue_at_{libtime.get_time()}")

    # -- 7. Blank gaze period (2000 ms) ---------------------------------------
    t0_blank = draw_blank(win)
    if tracker:
        tracker.log(f"{tag}_StartBlankGaze_cued_{img_for_gaze}_at_{t0_blank}")
    wait_ms(T_BLANK_GAZE)
    if tracker:
        tracker.log(f"{tag}_EndBlankGaze_at_{libtime.get_time()}")

    # -- 8. Vividness rating (blocks until keypress 1-4) ----------------------
    if tracker:
        tracker.log(f"{tag}_StartVividnessRating")
    vividness = get_vividness(win)
    if tracker:
        tracker.log(f"{tag}_VividnessRating_{vividness}")

    # -- 9. ITI blank (1000 ms) -----------------------------------------------
    draw_blank(win)
    wait_ms(T_ITI)

    # -- ET: stop recording & log variables -----------------------------------
    cued_image = img1 if img_for_gaze == 1 else img2
    if tracker:
        tracker.stop_recording()
        tracker.log_var("trial_num",             trial_num)
        tracker.log_var("condition_label",       condition_label)
        tracker.log_var("img_first",             img1)
        tracker.log_var("img_second",            img2)
        tracker.log_var("ImgForGaze_1st_or_2nd", img_for_gaze)
        tracker.log_var("cued_image",            cued_image)
        tracker.log_var("vividness",             vividness)

    # -- CSV log (main trials only) -------------------------------------------
    if not is_training:
        log_rows.append({
            "trial_num":             trial_num,
            "condition_label":       condition_label,
            "img_first":             img1,
            "img_second":            img2,
            "ImgForGaze_1st_or_2nd": img_for_gaze,
            "cued_image":            cued_image,
            "vividness":             vividness,
        })

    return vividness


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

    # -- Build main trial list ------------------------------------------------
    all_trials = build_trial_list()
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

    # -- Final save -----------------------------------------------------------
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
