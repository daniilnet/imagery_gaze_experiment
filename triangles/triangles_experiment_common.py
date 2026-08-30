"""
triangles_experiment_common.py

Shared configuration, PsychoPy/PyGaze setup, and trial-running logic used by
both triangles_imagery.py (imagery task) and triangles_perception.py
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
        up.png      down.png

Eye tracker: Gazepoint (OpenGaze protocol).
"""

import os
import csv
import ctypes
from datetime import datetime

# Must run before any window is created, or Windows display scaling makes
# stimuli render far larger than SCREEN_W/SCREEN_H intend.
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI aware
except (AttributeError, OSError):
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass

# Must be set before any pygaze imports.
os.environ["DISPTYPE"] = "psychopy"
os.environ["TRACKERTYPE"] = "opengaze"

from psychopy import visual, core, event

import pygaze
import pygaze.settings as pygaze_settings
from pygaze.display import Display
from pygaze.eyetracker import EyeTracker
from pygaze import libtime

SCREEN_W    = 1920
SCREEN_H    = 1080   # change on lab computer
FULLSCREEN  = True
FG_COLOR    = "white"
CUE_COLOR   = (180, 180, 180)  # light gray, rgb255
POOL_DIR    = os.path.join(os.path.dirname(__file__), "..", "pool")
IMAGE_SCALE = 1.0   # stimuli are already 1920x1080 -- fill the screen as-is

UP_IMAGE    = "up.png"
DOWN_IMAGE  = "down.png"

# Timing (ms)
T_ITI            = 1000   # inter-trial interval, start and end of each trial
T_FIXATION       = 1500   # fixation cross
T_CUE            = 300    # up/down arrow cue, center of screen (imagery only)
T_IMAGERY_BLANK  = 3000   # blank imagery period (imagery mode)
T_PERCEPTION_IMG = 3000   # cued image display duration (perception mode)
T_INTRO_IMG      = 1500   # each two-stimulus preview image on screen
T_INTRO_BLANK    = 1000   # blank between the two preview images
T_TAG_GAP        = 30     # lets EndStep4 land in a gaze sample (150Hz) before
                          # being overwritten by StartVividnessRating


def load_trials(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows:
        row["trial_num"] = int(row["trial_num"])
    return rows


def pool(filename):
    return os.path.join(POOL_DIR, filename)


def wait_ms(ms):
    # hogCPUperiod=0.2: hogging the whole wait starves the OpenGaze socket
    # threads (and thus gaze samples) by holding the GIL.
    core.wait(ms / 1000.0, hogCPUperiod=0.2)


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


def run_trial_sequence(win, tracker, trial_num, trial_def,
                       log_rows, is_training=False, mode='imagery'):
    """Runs one full trial. tracker=None skips all ET calls (training)."""
    cue          = trial_def["cue"]
    cued_image   = trial_def["cued_image"]
    first_image  = trial_def["first_image"]
    second_image = trial_def["second_image"]
    tag          = "training" if is_training else f"{mode}_{trial_num}"

    # 0. Two-stimulus preview (counterbalanced order)
    show_trial_intro(win, tracker, tag, first_image, second_image)

    # 1. ITI blank, start
    draw_blank(win)
    wait_ms(T_ITI)

    if tracker:
        tracker.start_recording()

    # 2. Fixation cross
    t0_fix = draw_cross(win)
    if tracker:
        tracker.log(f"{tag}_StartFixation_at_{t0_fix}")
    wait_ms(T_FIXATION)
    if tracker:
        tracker.log(f"{tag}_EndFixation_at_{libtime.get_time()}")

    # 3. Up/down arrow cue (imagery only)
    if mode != 'perception':
        t0 = draw_text(win, cue, height=60, color=CUE_COLOR, color_space='rgb255')
        if tracker:
            tracker.log(f"{tag}_StartCue_{cue}_at_{t0}")
        wait_ms(T_CUE)
        if tracker:
            tracker.log(f"{tag}_EndCue_at_{libtime.get_time()}")

    # 4. Blank imagery period (imagery) or cued image (perception)
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
        wait_ms(T_TAG_GAP)

    # 5-6. Ratings (imagery only, keypress 1-4)
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

    # 7. ITI blank, end
    draw_blank(win)

    # log_var() -> tracker.log() only reaches the gaze file via the sample
    # stream, and is skipped entirely once recording is False -- so log
    # variables here, before stop_recording(), or they're silently dropped.
    if tracker:
        tracker.log_var("phase",        mode)
        tracker.log_var("trial_num",    trial_num)
        tracker.log_var("cue",          cue)
        tracker.log_var("cued_image",   cued_image)
        tracker.log_var("first_image",  first_image)
        tracker.log_var("second_image", second_image)
        if vividness is not None:
            tracker.log_var("vividness",         vividness)
        if time_to_imagine is not None:
            tracker.log_var("time_to_imagine",   time_to_imagine)

    wait_ms(T_ITI)

    if tracker:
        tracker.stop_recording()

    if not is_training:
        row = {
            "phase":           mode,
            "trial_num":       trial_num,
            "cue":             cue,
            "cued_image":      cued_image,
            "first_image":     first_image,
            "second_image":    second_image,
            "vividness":       vividness,
            "time_to_imagine": time_to_imagine,
        }
        log_rows.append(row)


def run_trials(win, tracker, trials, log_rows, start_trial_num,
               mode='imagery', is_training=False,
               break_every=None, disp=None, log_file=None, without_tracker=False):
    """Runs `trials` in order, with a break screen every `break_every` trials."""
    trial_num = start_trial_num

    for i, trial_def in enumerate(trials, start=1):
        run_trial_sequence(win, tracker, trial_num, trial_def,
                           log_rows, is_training=is_training, mode=mode)
        trial_num += 1

        if break_every and i % break_every == 0 and i != len(trials):
            break_screen(win, tracker, disp, log_rows, log_file, without_tracker)

    return trial_num


def run_training(win, training_trials, mode='imagery'):
    run_trials(win, tracker=None, trials=training_trials, log_rows=[],
               start_trial_num=1, is_training=True, mode=mode)
    draw_blank(win)
    event.clearEvents()
    core.wait(0.5)


def quit_and_save(win, tracker, disp, log_rows, log_file, without_tracker):
    save_csv(log_rows, log_file, without_tracker)
    if tracker:
        tracker.close()
    disp.close()
    win.close()
    core.quit()


def break_screen(win, tracker, disp, log_rows, log_file, without_tracker):
    """Break screen: saves data, Space continues, Q saves and quits."""
    save_csv(log_rows, log_file, without_tracker)
    draw_text(win, "Take a short break.\n\nPress SPACE to continue.")
    key = wait_keypress(win, keys=['space', 'q'])
    if key == 'q':
        quit_and_save(win, tracker, disp, log_rows, log_file, without_tracker)


def wait_start_keypress(win, tracker, disp, log_rows, log_file, without_tracker):
    """Like wait_keypress(['space']), but Q saves and quits."""
    key = wait_keypress(win, keys=['space', 'q'])
    if key == 'q':
        quit_and_save(win, tracker, disp, log_rows, log_file, without_tracker)
    return key


def prompt_subject_number(without_tracker=False):
    """Prompts for a subject number, or in dev mode just confirms to continue."""
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


def setup_display_and_tracker(without_tracker, et_log):
    # Let PyGaze own the window (prevents double-window on startup).
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
