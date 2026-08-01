"""
multi_stim_experiment_common.py

Shared configuration, PsychoPy/PyGaze setup, and trial-running logic used by
both multi_stim_imagery.py (imagery task) and multi_stim_perception.py
(perception task). Neither task script imports the other -- they are run as
separate sessions.

Dependencies:
    requires python 3.10.11 specifically!
    downloaded from https://www.python.org/downloads/windows/

    pip install psychopy lxml pygame
    pip install https://github.com/esdalmaijer/PyGaze/archive/refs/heads/master.zip
    (notice there are different 'pygaze' packages, download from the address above)

Folder structure expected (relative to the project root, one level up from
this file):
    pool/
        face1_left.png   face1_right.png   ...   face4_left.png   face4_right.png
        house1_left.png  house1_right.png  ...   house4_left.png  house4_right.png

    Each face/house identity (1-4) has a "_left" and "_right" render (the
    picture-frame content sits on the left or right side of the same room
    scene). Which identity and which side is used for each stimulus on each
    trial is fixed per-participant by the trial CSVs (see
    generate_trial_csvs.py) and counterbalanced across the session.

Eye tracker: Gazepoint (OpenGaze protocol).

Trial sequence (timings):
    0. Two-stimulus preview (every trial, counterbalanced order and sides)
       first_image                     1500 ms
       blank                           1000 ms
       second_image                    1500 ms
    1. ITI blank (start)               1000 ms
    2. Fixation cross                  1500 ms
    3. H/F cue (center of screen)       300 ms   (imagery only)
    4. Blank imagery period            3000 ms   (imagery)
       Cued image (house/face)         3000 ms   (perception)
    5. Vividness rating (1-4)         until keypress   (imagery only)
    6. Time-to-imagine rating (1-4)   until keypress   (imagery only)
    7. ITI blank (end)                 1000 ms
"""

import os
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
BACKGROUND  = "#000000"
FG_COLOR    = "white"
CUE_COLOR   = (180, 180, 180)  # light gray, rgb255
POOL_DIR    = os.path.join(os.path.dirname(__file__), "..", "pool")
IMAGE_SCALE = 1.5

# Timing (ms)
T_ITI            = 1000   # inter-trial interval, at both start and end of each trial
T_FIXATION       = 1500   # fixation cross
T_CUE            = 300    # H/F cue in center of screen (imagery mode only)
T_IMAGERY_BLANK  = 3000   # blank imagery period (imagery mode)
T_PERCEPTION_IMG = 3000   # cued image display duration (perception mode)
T_INTRO_IMG      = 1500   # each two-stimulus preview image on screen
T_INTRO_BLANK    = 1000   # blank between the two preview images


# -----------------------------------------------------------------------------
# TRIAL CSV LOADING
# -----------------------------------------------------------------------------
def load_trials(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows:
        row["trial_num"] = int(row["trial_num"])
        row["face_id"]   = int(row["face_id"])
        row["house_id"]  = int(row["house_id"])
    return rows


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


def draw_text(win, text, height=30, color=FG_COLOR, color_space='rgb'):
    stim = visual.TextStim(win, text=text, color=color, colorSpace=color_space, height=height,
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


def save_csv(log_rows, log_file, without_tracker):
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
    "How vivid was your imagery?"
)

TIME_TO_IMAGINE_PROMPT = (
    "When did the image come to mind?"
)


def get_rating(win, prompt_text):
    draw_text(win, prompt_text)
    key = wait_keypress(win, keys=['1', '2', '3', '4'])
    return int(key)


# -----------------------------------------------------------------------------
# TWO-STIMULUS PREVIEW (shown at the start of every trial, counterbalanced
# order given by the trial definition's first_image / second_image)
# -----------------------------------------------------------------------------
def show_trial_intro(win, tracker, tag, first_image, second_image):
    t0 = draw_image(win, first_image)
    if tracker:
        tracker.log(f"{tag}_TrialIntro_{first_image}_at_{t0}")
    wait_ms(T_INTRO_IMG)

    draw_blank(win)
    wait_ms(T_INTRO_BLANK)

    t0 = draw_image(win, second_image)
    if tracker:
        tracker.log(f"{tag}_TrialIntro_{second_image}_at_{t0}")
    wait_ms(T_INTRO_IMG)


# -----------------------------------------------------------------------------
# CORE TRIAL SEQUENCE
# -----------------------------------------------------------------------------
def run_trial_sequence(win, tracker, trial_num, trial_def,
                       log_rows, is_training=False, mode='imagery'):
    """
    Runs one full trial, including the two-stimulus preview.
    tracker=None  -> skips all ET calls (training).
    mode='imagery'    -> step 4 is a blank imagery period.
    mode='perception' -> step 4 shows the cued image.
    Returns nothing; appends a row to log_rows (unless is_training).
    """
    cue          = trial_def["cue"]
    cued_image   = trial_def["cued_image"]
    cued_side    = trial_def["cued_side"]
    face_id      = trial_def["face_id"]
    face_side    = trial_def["face_side"]
    face_image   = trial_def["face_image"]
    house_id     = trial_def["house_id"]
    house_side   = trial_def["house_side"]
    house_image  = trial_def["house_image"]
    first_image  = trial_def["first_image"]
    second_image = trial_def["second_image"]
    tag          = "training" if is_training else f"{mode}_{trial_num}"

    # -- 0. Two-stimulus preview (counterbalanced order) -----------------------
    show_trial_intro(win, tracker, tag, first_image, second_image)

    # -- 1. ITI blank, start (1000 ms) ----------------------------------------
    draw_blank(win)
    wait_ms(T_ITI)

    if tracker:
        tracker.start_recording()

    # -- 2. Fixation cross (1500 ms) -------------------------------------------
    t0_fix = draw_cross(win)
    if tracker:
        tracker.log(f"{tag}_StartFixation_at_{t0_fix}")
    wait_ms(T_FIXATION)
    if tracker:
        tracker.log(f"{tag}_EndFixation_at_{libtime.get_time()}")

    # -- 3. H/F cue in center of screen (300 ms) -- imagery only ---------------
    if mode != 'perception':
        t0 = draw_text(win, cue, height=60, color=CUE_COLOR, color_space='rgb255')
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
        tracker.log_var("phase",        mode)
        tracker.log_var("trial_num",    trial_num)
        tracker.log_var("cue",          cue)
        tracker.log_var("cued_image",   cued_image)
        tracker.log_var("cued_side",    cued_side)
        tracker.log_var("face_id",      face_id)
        tracker.log_var("face_side",    face_side)
        tracker.log_var("face_image",   face_image)
        tracker.log_var("house_id",     house_id)
        tracker.log_var("house_side",   house_side)
        tracker.log_var("house_image",  house_image)
        tracker.log_var("first_image",  first_image)
        tracker.log_var("second_image", second_image)
        if vividness is not None:
            tracker.log_var("vividness",         vividness)
        if time_to_imagine is not None:
            tracker.log_var("time_to_imagine",   time_to_imagine)

    # -- CSV log (non-training trials only) -----------------------------------
    if not is_training:
        row = {
            "phase":           mode,
            "trial_num":       trial_num,
            "cue":             cue,
            "cued_image":      cued_image,
            "cued_side":       cued_side,
            "face_id":         face_id,
            "face_side":       face_side,
            "face_image":      face_image,
            "house_id":        house_id,
            "house_side":      house_side,
            "house_image":     house_image,
            "first_image":     first_image,
            "second_image":    second_image,
            "vividness":       vividness,
            "time_to_imagine": time_to_imagine,
        }
        log_rows.append(row)


# -----------------------------------------------------------------------------
# RUN A FLAT LIST OF TRIALS, WITH AN OPTIONAL BREAK EVERY N TRIALS
# -----------------------------------------------------------------------------
def run_trials(win, tracker, trials, log_rows, start_trial_num,
               mode='imagery', is_training=False,
               break_every=None, disp=None, log_file=None, without_tracker=False):
    """
    Runs `trials` (a flat list of trial_def dicts) in order. If break_every
    is set, shows a break screen after every N trials (except after the
    final trial).
    """
    trial_num = start_trial_num

    for i, trial_def in enumerate(trials, start=1):
        run_trial_sequence(win, tracker, trial_num, trial_def,
                           log_rows, is_training=is_training, mode=mode)
        trial_num += 1

        if break_every and i % break_every == 0 and i != len(trials):
            break_screen(win, tracker, disp, log_rows, log_file, without_tracker)

    return trial_num


# -----------------------------------------------------------------------------
# TRAINING SESSION
# -----------------------------------------------------------------------------
def run_training(win, training_trials, mode='imagery'):
    run_trials(win, tracker=None, trials=training_trials, log_rows=[],
               start_trial_num=1, is_training=True, mode=mode)
    draw_blank(win)
    event.clearEvents()
    core.wait(0.5)


# -----------------------------------------------------------------------------
# EXPERIMENTER QUIT  (save data and close everything down)
# -----------------------------------------------------------------------------
def quit_and_save(win, tracker, disp, log_rows, log_file, without_tracker):
    save_csv(log_rows, log_file, without_tracker)
    if tracker:
        tracker.close()
    disp.close()
    win.close()
    core.quit()


# -----------------------------------------------------------------------------
# BREAK SCREEN  (saves data; shows space to continue, accepts q to quit)
# -----------------------------------------------------------------------------
def break_screen(win, tracker, disp, log_rows, log_file, without_tracker):
    """
    Shows a break screen. Saves data automatically.
    Participant sees only the space-to-continue prompt.
    Experimenter can press Q to save and quit.
    """
    save_csv(log_rows, log_file, without_tracker)
    draw_text(win, "Take a short break.\n\nPress SPACE to continue.")
    key = wait_keypress(win, keys=['space', 'q'])
    if key == 'q':
        quit_and_save(win, tracker, disp, log_rows, log_file, without_tracker)


# -----------------------------------------------------------------------------
# START-OF-EXPERIMENT PROMPT  (space to continue, accepts q to save & quit)
# -----------------------------------------------------------------------------
def wait_start_keypress(win, tracker, disp, log_rows, log_file, without_tracker):
    """
    Like wait_keypress(keys=['space']), but the experimenter can press Q to
    silently save whatever has been logged so far and quit. Participant sees
    only the space-to-continue prompt already drawn on screen.
    """
    key = wait_keypress(win, keys=['space', 'q'])
    if key == 'q':
        quit_and_save(win, tracker, disp, log_rows, log_file, without_tracker)
    return key


# -----------------------------------------------------------------------------
# SUBJECT / LOG-FILE SETUP
# -----------------------------------------------------------------------------
def prompt_subject_number(without_tracker=False):
    """
    Normally prompts for and returns an integer subject number. In dev mode
    (without_tracker=True), skips that prompt entirely -- no data will be
    saved anyway -- and instead warns the experimenter and asks for
    confirmation to continue.
    """
    if without_tracker:
        print("\n*** DEV MODE (without_tracker=1): no data will be saved. ***")
        while True:
            answer = input("Continue anyway? (y/n): ").strip().lower()
            if answer in ("y", "yes"):
                return "dev"
            if answer in ("n", "no"):
                print("Exiting.")
                core.quit()
            print("Please enter y or n.")

    while True:
        try:
            return int(input("Enter subject number: "))
        except ValueError:
            print("Please enter a valid integer.")


def make_log_paths(task_name, subject_nr):
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(results_dir, f"log_{task_name}_subj_{subject_nr}_{timestamp}.csv")
    et_log   = os.path.join(results_dir, f"gazepoint_data_{task_name}_subj_{subject_nr}_{timestamp}")
    return log_file, et_log


# -----------------------------------------------------------------------------
# DISPLAY / EYE TRACKER SETUP
# -----------------------------------------------------------------------------
def setup_display_and_tracker(without_tracker, et_log):
    # -- Let PyGaze own the window (prevents double-window on startup) --------
    pygaze_settings.DISPSIZE   = (SCREEN_W, SCREEN_H)
    pygaze_settings.SCREENNR   = 0
    pygaze_settings.FULLSCREEN = FULLSCREEN
    pygaze_settings.BGC        = (0, 0, 0)  # pure black

    disp = Display()
    win  = pygaze.expdisplay  # window PyGaze created, stored on the module
    if without_tracker:
        tracker = None
        print("without_tracker=1: running WITHOUT eye tracker, no ET data will be logged.")
    else:
        tracker = EyeTracker(disp, trackertype="opengaze", logfile=et_log)
    return disp, win, tracker
