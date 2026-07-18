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
        face_right.png      house_right.png

Eye tracker: Gazepoint (OpenGaze protocol).

Design:
    Only the "right" stimuli are used (face_right.png / house_right.png).
    Trials are organized into blocks of 6. Each block is preceded by a
    single, fixed-order presentation of both stimuli (face_right then
    house_right) -- not randomized, not repeated within the block.

    Training:   1 block (intro images + 6 practice trials, no ET, not logged).

    Imagery:    10 blocks x 6 trials = 60 trials.
                Break screen after every 2 blocks (except after the last).

    Perception: 2 blocks x 6 trials = 12 trials, run after all imagery blocks.
                Same structure, but the cued image (matching the H/F cue) is
                shown during the imagery period instead of a blank screen;
                no vividness / time-to-imagine ratings are collected.

Trial sequence (timings):
    1. ITI blank (start)              1000 ms
    2. Fixation cross                  500 ms
    3. H/F cue (center of screen)      300 ms
    4. Blank imagery period           3000 ms   (imagery)
       Cued image (house/face)        3000 ms   (perception)
    5. Vividness rating (1-4)         until keypress   (imagery only)
    6. Time-to-imagine rating (1-4)   until keypress   (imagery only)
    7. ITI blank (end)                1000 ms

Block intro (before each block's first trial):
    face_right.png   2000 ms
    blank             500 ms
    house_right.png  2000 ms
"""

import os
import csv
from datetime import datetime

# -----------------------------------------------------------------------------
# DEV MODE SWITCH
# -----------------------------------------------------------------------------
without_tracker = 1   # 1 = run experiment without eye tracker connected (testing).
                      # On-screen is identical to normal runtime; no ET/gaze data is logged.


def _load_trials(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows:
        row["block"] = int(row["block"])
        row["trial_in_block"] = int(row["trial_in_block"])
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

FACE_IMAGE  = "face_right.png"
HOUSE_IMAGE = "house_right.png"

# Timing (ms)
T_ITI           = 1000   # inter-trial interval, at both start and end of each trial
T_FIXATION      = 1500    # fixation cross
T_CUE           = 300    # H/F cue in center of screen
T_IMAGERY_BLANK = 3000   # blank imagery period (imagery mode)
T_PERCEPTION_IMG = 3000  # cued image display duration (perception mode)
T_INTRO_IMG     = 3000   # each block-intro stimulus on screen
T_INTRO_BLANK   = 500    # blank between the two block-intro stimuli

BLOCK_SIZE          = 6   # trials per block
BREAK_EVERY_BLOCKS  = 2   # rest break after every N blocks (imagery only)

# -----------------------------------------------------------------------------
# FIXED PSEUDORANDOM TRIAL ORDERS (see generate_trial_csvs.py)
# Same H/F cue sequence for every participant; counterbalanced 3H/3F per
# block with no more than 2 identical cues in a row.
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
    if without_tracker:
        return
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
# BLOCK INTRO (both stimuli shown once, fixed order, before each block)
# -----------------------------------------------------------------------------
def show_block_intro(win, tracker, tag):
    t0 = draw_image(win, FACE_IMAGE)
    if tracker:
        tracker.log(f"{tag}_BlockIntro_{FACE_IMAGE}_at_{t0}")
    wait_ms(T_INTRO_IMG)

    draw_blank(win)
    wait_ms(T_INTRO_BLANK)

    t0 = draw_image(win, HOUSE_IMAGE)
    if tracker:
        tracker.log(f"{tag}_BlockIntro_{HOUSE_IMAGE}_at_{t0}")
    wait_ms(T_INTRO_IMG)


# -----------------------------------------------------------------------------
# CORE TRIAL SEQUENCE
# -----------------------------------------------------------------------------
def run_trial_sequence(win, tracker, trial_num, trial_def,
                       log_rows, is_training=False, mode='imagery'):
    """
    Runs one full trial.
    tracker=None  -> skips all ET calls (training).
    mode='imagery'    -> step 4 is a blank imagery period.
    mode='perception' -> step 4 shows the cued image.
    Returns nothing; appends a row to log_rows (unless is_training).
    """
    block          = trial_def["block"]
    trial_in_block = trial_def["trial_in_block"]
    cue            = trial_def["cue"]
    cued_image     = trial_def["cued_image"]
    tag            = "training" if is_training else f"{mode}_{trial_num}"

    # -- 1. ITI blank, start (1000 ms) ----------------------------------------
    draw_blank(win)
    wait_ms(T_ITI)

    if tracker:
        tracker.start_recording()

    # -- 2. Fixation cross (500 ms) -------------------------------------------
    t0_fix = draw_cross(win)
    if tracker:
        tracker.log(f"{tag}_StartFixation_at_{t0_fix}")
    wait_ms(T_FIXATION)
    if tracker:
        tracker.log(f"{tag}_EndFixation_at_{libtime.get_time()}")

    # -- 3. H/F cue in center of screen (300 ms) -------------------------------
    t0 = draw_text(win, cue, height=60)
    if tracker:
        tracker.log(f"{tag}_StartCue_{cue}_at_{t0}")
    wait_ms(T_CUE)
    if tracker:
        tracker.log(f"{tag}_EndCue_at_{libtime.get_time()}")

    # -- 4. Blank imagery period (imagery) or cued image (perception) ---------
    if mode == 'perception':
        t0 = draw_image(win, cued_image)
        if tracker:
            tracker.log(f"{tag}_StartPerceptionImage_{cued_image}_at_{t0}")
        wait_ms(T_PERCEPTION_IMG)
    else:
        t0 = draw_blank(win)
        if tracker:
            tracker.log(f"{tag}_StartImageryBlank_cued_{cue}_at_{t0}")
        wait_ms(T_IMAGERY_BLANK)
    if tracker:
        tracker.log(f"{tag}_EndStep4_at_{libtime.get_time()}")

    # -- 5-6. Ratings (imagery only, keypress 1-4) -----------------------------
    vividness = None
    time_to_imagine = None
    if mode != 'perception':
        if tracker:
            tracker.log(f"{tag}_StartVividnessRating")
        vividness = get_rating(win, VIVIDNESS_PROMPT)
        if tracker:
            tracker.log(f"{tag}_VividnessRating_{vividness}")

        if tracker:
            tracker.log(f"{tag}_StartTimeToImagineRating")
        time_to_imagine = get_rating(win, TIME_TO_IMAGINE_PROMPT)
        if tracker:
            tracker.log(f"{tag}_TimeToImagineRating_{time_to_imagine}")

    # -- 7. ITI blank, end (1000 ms) -------------------------------------------
    draw_blank(win)
    wait_ms(T_ITI)

    # -- ET: stop recording & log variables -----------------------------------
    if tracker:
        tracker.stop_recording()
        tracker.log_var("phase",           mode)
        tracker.log_var("trial_num",       trial_num)
        tracker.log_var("block",           block)
        tracker.log_var("trial_in_block",  trial_in_block)
        tracker.log_var("cue",             cue)
        tracker.log_var("cued_image",      cued_image)
        if vividness is not None:
            tracker.log_var("vividness",         vividness)
        if time_to_imagine is not None:
            tracker.log_var("time_to_imagine",   time_to_imagine)

    # -- CSV log (non-training trials only) -----------------------------------
    if not is_training:
        row = {
            "phase":           mode,
            "trial_num":       trial_num,
            "block":           block,
            "trial_in_block":  trial_in_block,
            "cue":             cue,
            "cued_image":      cued_image,
            "vividness":       vividness,
            "time_to_imagine": time_to_imagine,
        }
        log_rows.append(row)


# -----------------------------------------------------------------------------
# RUN A PHASE (training / imagery / perception), block by block
# -----------------------------------------------------------------------------
def run_blocks(win, tracker, trials, log_rows, start_trial_num,
               mode='imagery', is_training=False,
               break_every_blocks=None, disp=None, log_file=None):
    """
    Runs `trials` (a flat list of trial_def dicts already tagged with
    block/trial_in_block) in blocks of BLOCK_SIZE, showing the block intro
    (both stimuli, fixed order) before each block.

    If break_every_blocks is set, shows a break screen after every N blocks
    (except after the final block).
    """
    trial_num = start_trial_num
    blocks = [trials[i:i + BLOCK_SIZE] for i in range(0, len(trials), BLOCK_SIZE)]

    for block_idx, block_trials in enumerate(blocks, start=1):
        tag = "training" if is_training else f"{mode}_block{block_idx}"
        show_block_intro(win, tracker, tag)

        for trial_def in block_trials:
            run_trial_sequence(win, tracker, trial_num, trial_def,
                               log_rows, is_training=is_training, mode=mode)
            trial_num += 1

        if (break_every_blocks and block_idx % break_every_blocks == 0
                and block_idx != len(blocks)):
            break_screen(win, tracker, disp, log_rows, log_file)

    return trial_num


# -----------------------------------------------------------------------------
# TRAINING SESSION
# -----------------------------------------------------------------------------
def run_training(win):
    run_blocks(win, tracker=None, trials=FIXED_TRAINING_TRIALS, log_rows=[],
               start_trial_num=1, is_training=True)
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
        if tracker:
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
    if without_tracker:
        tracker = None
        print("without_tracker=1: running WITHOUT eye tracker, no ET data will be logged.")
    else:
        tracker = EyeTracker(disp, trackertype="opengaze", logfile=et_log)

    # -- Training -------------------------------------------------------------
    draw_text(win, "Training\n\nPress SPACE to begin.")
    wait_keypress(win, keys=['space'])
    run_training(win)

    # -- Experiment start -----------------------------------------------------
    draw_text(win, "Experiment\n\nPress SPACE to begin.")
    wait_keypress(win, keys=['space'])

    # -- Imagery blocks, with breaks every BREAK_EVERY_BLOCKS blocks -----------
    log_rows  = []
    trial_num = run_blocks(win, tracker, FIXED_IMAGERY_TRIALS, log_rows,
                           start_trial_num=1, mode='imagery',
                           break_every_blocks=BREAK_EVERY_BLOCKS,
                           disp=disp, log_file=log_file)

    # -- Save after imagery block ---------------------------------------------
    save_csv(log_rows, log_file)

    # =========================================================================
    # PERCEPTION
    # =========================================================================
    draw_text(win, "Perception section\n\nPress SPACE to begin.")
    wait_keypress(win, keys=['space'])

    run_blocks(win, tracker, FIXED_PERCEPTION_TRIALS, log_rows,
              start_trial_num=trial_num, mode='perception',
              break_every_blocks=None, disp=disp, log_file=log_file)

    # -- Final save (imagery + perception) ------------------------------------
    save_csv(log_rows, log_file)

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
