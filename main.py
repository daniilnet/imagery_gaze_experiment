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

    Imagery:    4 blocks x 12 trials = 48 trials.
                Each condition: one face + one house image.
                Break every 12 trials (3 breaks total within imagery block).

    Perception: 12 trials.
                Same as imagery but step 7 shows the cued image instead of blank.

Trial sequence (timings):
    1. Start fixation cross          1500 ms
    2. Image 1                       2000 ms
    3. Blank (no cross)               500 ms
    4. Image 2                       2000 ms
    5. Blank after image 2            500 ms
    6. Retrocue number                300 ms
    7. Blank gaze                    3500 ms   (imagery)
       Blank 200ms + cued image     3500 ms   (perception)
    8. Vividness rating (1-4)        until keypress
    9. Time-to-imagine rating (1-4)  until keypress
   10. ITI blank                     1500 ms
"""

import os
import csv
from datetime import datetime


def _load_trials(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows:
        row["img_for_gaze"] = int(row["img_for_gaze"])
    return rows

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
BACKGROUND  = "#000000"
FG_COLOR    = "white"
POOL_DIR    = os.path.join(os.path.dirname(__file__), "pool")
IMAGE_SCALE = 1.5

# Timing (ms)
T_START_FIX     = 1500   # fixation cross before first image
T_IMG           = 2000   # each image on screen
T_BETWEEN_BLANK = 500    # blank between images (no cross)
T_POST_IMG2     = 500    # blank after image 2, replaces mask
T_RETROCUE      = 300    # retrocue number
T_BLANK_GAZE    = 3500   # blank gaze period
T_ITI           = 1500   # inter-trial interval

BREAK_EVERY = 12         # show a break screen after every N imagery trials

# -- Experiment design --------------------------------------------------------
# Each entry: (img_first, img_second, condition_label)
CONDITIONS = [
    ("face_right.png",  "house_left.png",  "face_R_house_L_face1st"),
    ("face_left.png",   "house_right.png", "face_L_house_R_face1st"),
    ("house_right.png", "face_left.png",   "house_R_face_L_house1st"),
    ("house_left.png",  "face_right.png",  "house_L_face_R_house1st"),
]
TRIALS_PER_CONDITION      = 12   # 4 blocks x 12 = 48 imagery trials total
N_TRAINING                = 4    # one per condition
PERCEPTION_TOTAL          = 12   # 12 perception trials total

# -----------------------------------------------------------------------------
# FIXED PSEUDORANDOM TRIAL ORDERS  (seed=42, same for every participant)
# -----------------------------------------------------------------------------
FIXED_IMAGERY_TRIALS    = _load_trials("imagery_trials.csv")
FIXED_TRAINING_TRIALS   = _load_trials("training_trials.csv")
FIXED_PERCEPTION_TRIALS = _load_trials("perception_trials.csv")

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
    if size is None:
        from PIL import Image as _PIL
        w, h = _PIL.open(pool(filename)).size
        size = (int(w * IMAGE_SCALE), int(h * IMAGE_SCALE))
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
# RATINGS (keypress 1-4)
# -----------------------------------------------------------------------------
VIVIDNESS_PROMPT = (
    "How vivid was your imagery?\n\n"
    "1 - Not vivid at all\n"
    "2 - Somewhat vivid\n"
    "3 - Vivid\n"
    "4 - Very vivid"
)

TIME_TO_IMAGINE_PROMPT = (
    "When did the image come to mind?\n\n"
    "1 - Immediately\n"
    "2 - Before the half-time\n"
    "3 - After the half-time\n"
    "4 - Near the end"
)


def get_rating(win, prompt_text):
    draw_text(win, prompt_text)
    key = wait_keypress(win, keys=['1', '2', '3', '4'])
    return int(key)


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
        draw_blank(win)
        wait_ms(200)
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

    # -- 8. Vividness rating (imagery only, keypress 1-4) ----------------------
    vividness = None
    time_to_imagine = None
    if mode != 'perception':
        if tracker:
            tracker.log(f"{tag}_StartVividnessRating")
        vividness = get_rating(win, VIVIDNESS_PROMPT)
        if tracker:
            tracker.log(f"{tag}_VividnessRating_{vividness}")

        # -- 9. Time-to-imagine rating (keypress 1-4) -------------------------
        if tracker:
            tracker.log(f"{tag}_StartTimeToImagineRating")
        time_to_imagine = get_rating(win, TIME_TO_IMAGINE_PROMPT)
        if tracker:
            tracker.log(f"{tag}_TimeToImagineRating_{time_to_imagine}")

    # -- 10. ITI blank (1500 ms) ----------------------------------------------
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
        if time_to_imagine is not None:
            tracker.log_var("time_to_imagine",   time_to_imagine)

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
            "time_to_imagine":       time_to_imagine,
        }
        log_rows.append(row)


# -----------------------------------------------------------------------------
# TRAINING SESSION
# -----------------------------------------------------------------------------
def run_training(win):
    for i, trial_def in enumerate(FIXED_TRAINING_TRIALS, start=1):
        run_trial_sequence(win, tracker=None, trial_num=i,
                           trial_def=trial_def, log_rows=[],
                           is_training=True)
    draw_blank(win)
    event.clearEvents()
    core.wait(0.5)


# -----------------------------------------------------------------------------
# BREAK SCREEN  (saves data; shows space to continue, accepts q to quit)
# -----------------------------------------------------------------------------
def break_screen(win, tracker, disp, log_rows, log_file):
    """
    Shows a break screen. Saves data automatically.
    Participant sees only the space-to-continue prompt.
    Experimenter can press Q to save and quit.
    Returns True to continue, False to quit.
    """
    save_csv(log_rows, log_file)
    draw_text(win, "Take a short break.\n\nPress SPACE to continue.")
    key = wait_keypress(win, keys=['space', 'q'])
    if key == 'q':
        tracker.close()
        disp.close()
        win.close()
        core.quit()


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

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file  = os.path.join(results_dir, f"log_subj_{subject_nr}_{timestamp}.csv")
    et_log    = os.path.join(results_dir, f"gazepoint_data_subj_{subject_nr}_{timestamp}")

    # -- Let PyGaze own the window (prevents double-window on startup) --------
    pygaze_settings.DISPSIZE   = (SCREEN_W, SCREEN_H)
    pygaze_settings.SCREENNR   = 0
    pygaze_settings.FULLSCREEN = FULLSCREEN
    pygaze_settings.BGCOLOUR   = (127, 127, 127)  # standard gray

    disp    = Display()
    win     = pygaze.expdisplay  # window PyGaze created, stored on the module
    tracker = EyeTracker(disp, trackertype="opengaze", logfile=et_log)

    # -- Training -------------------------------------------------------------
    draw_text(win, "Training\n\nPress SPACE to begin.")
    wait_keypress(win, keys=['space'])
    run_training(win)

    # -- Experiment start -----------------------------------------------------
    draw_text(win, "Experiment\n\nPress SPACE to begin.")
    wait_keypress(win, keys=['space'])

    # -- Imagery trials with breaks every BREAK_EVERY trials ------------------
    log_rows  = []
    trial_num = 1
    total     = len(FIXED_IMAGERY_TRIALS)

    for chunk_start in range(0, total, BREAK_EVERY):
        chunk = FIXED_IMAGERY_TRIALS[chunk_start:chunk_start + BREAK_EVERY]
        for trial_def in chunk:
            run_trial_sequence(win, tracker, trial_num, trial_def, log_rows)
            trial_num += 1

        # Break after every chunk except the last
        if chunk_start + BREAK_EVERY < total:
            break_screen(win, tracker, disp, log_rows, log_file)

    # -- Save after imagery block ---------------------------------------------
    save_csv(log_rows, log_file)

    # =========================================================================
    # PERCEPTION
    # =========================================================================
    draw_text(win, "Perception section\n\nPress SPACE to begin.")
    wait_keypress(win, keys=['space'])

    for trial_def in FIXED_PERCEPTION_TRIALS:
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
