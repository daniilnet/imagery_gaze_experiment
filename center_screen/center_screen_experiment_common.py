import csv
import ctypes
import os
import time
from datetime import datetime

# Make this process DPI-aware BEFORE any window is created
# Without this, Windows display scaling (e.g. 125%/150% on a laptop) reports a
# smaller virtual resolution to the process than the panel's real pixel size,
# so the hardcoded SCREEN_W/SCREEN_H below no longer matches the window
# PsychoPy actually creates, and stimuli render far larger than intended.
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI aware
except (AttributeError, OSError):
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass

os.environ["DISPTYPE"] = "psychopy"
os.environ["TRACKERTYPE"] = "opengaze"

import pygaze
import pygaze.settings as pygaze_settings
from psychopy import core, event, logging, visual
from pygaze import libtime
from pygaze.display import Display
from pygaze.eyetracker import EyeTracker

SCREEN_W    = 1920
SCREEN_H    = 1080   # change on lab computer
FULLSCREEN  = True
FG_COLOR    = "white"
CUE_COLOR   = (180, 180, 180)  # light gray, rgb255
POOL_DIR    = os.path.join(os.path.dirname(__file__), "..", "pool")
IMAGE_SCALE = 0.3

FACE_IMAGE  = "face_center.png"
HOUSE_IMAGE = "house_center.png"
CUES        = ("F", "H")   # every cue letter that can appear in a trial CSV

# Per-image vertical nudge (pix, +up/-down) applied on top of screen-center
# positioning -- lets the face/house be fine-tuned independently since their
# "visual center of mass" differs from their pixel-center.
IMAGE_Y_OFFSET = {
    FACE_IMAGE:  -20,  # move down (px)
    HOUSE_IMAGE:  20,  # move up (px)
}

# -----------------------------------------------------------------------------
# CLOCKS
# -----------------------------------------------------------------------------
# Two different clocks feed the ET markers, and they are NOT interchangeable:
#     win.flip()         -> SECONDS,      psychopy.logging.defaultClock epoch
#     libtime.get_time() -> MILLISECONDS, pygaze expbegintime epoch
# Logging one of each (Start from the flip, End from libtime) left every marker
# pair 1000x apart in units and offset by an epoch difference that was never
# recorded, so "End - Start" was meaningless and not recoverable afterwards.
#
# Sample both clocks here, microseconds apart, and keep the offset. flip
# timestamps can then be expressed on libtime's millisecond timeline, so every
# "_at_" value in the gaze log shares one unit and one epoch.
_t_flip             = logging.defaultClock.getTime()
_FLIP_TO_LIBTIME_MS = libtime.get_time() - _t_flip * 1000.0
del _t_flip


def flip_time_ms(t_flip):
    """A win.flip() timestamp (s) expressed on libtime's millisecond clock."""
    return t_flip * 1000.0 + _FLIP_TO_LIBTIME_MS


# Timing (ms)
T_ITI            = 1000   # inter-trial interval, at both start and end of each trial
T_PRECUE_BLANK   = 250    # blank between the trial-intro preview and the H/F cue
T_CUE            = 300    # H/F cue in center of screen (imagery mode only)
T_IMAGERY_BLANK  = 4000   # blank imagery period (imagery mode)
T_PERCEPTION_IMG = 4000   # cued image display duration (perception mode)
T_INTRO_IMG      = 1500   # each two-stimulus preview image on screen
T_INTRO_BLANK    = 1000   # blank between the two preview images


def load_trials(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows:
        row["trial_num"] = int(row["trial_num"])
    return rows


def pool(filename):
    return os.path.join(POOL_DIR, filename)


def wait_ms(ms):
    # Hog the CPU only for the last 200 ms, as PsychoPy recommends. Hogging the
    # whole interval spins core.wait's tight loop (which re-parses the pyglet
    # version and pumps the window event queue on every pass) while holding the
    # GIL, starving the OpenGaze client's background socket threads -- and those
    # threads are what read samples off the tracker and queue them for the gaze
    # file, so starving them costs samples.
    core.wait(ms / 1000.0, hogCPUperiod=0.2)


# -----------------------------------------------------------------------------
# STIMULUS CACHE
# -----------------------------------------------------------------------------
# Building a PsychoPy stimulus is expensive: ImageStim reads the PNG off disk
# and uploads a texture, TextStim lays out and rasterises glyphs. Doing that
# inside a trial puts disk and GPU work between the draw call and the flip,
# which jitters the stimulus onset. So each stimulus is built once and then
# only re-drawn. preload_stimuli() forces the trial-critical set to be built at
# startup; everything else (instructions, break screens) is built on first use
# and cached from then on.
_SIZE_CACHE  = {}
_IMAGE_CACHE = {}
_TEXT_CACHE  = {}


def _scaled_size(filename):
    """Native PNG size * IMAGE_SCALE. Reads each file off disk at most once."""
    size = _SIZE_CACHE.get(filename)
    if size is None:
        from PIL import Image as _PIL
        with _PIL.open(pool(filename)) as im:
            w, h = im.size
        size = (int(w * IMAGE_SCALE), int(h * IMAGE_SCALE))
        _SIZE_CACHE[filename] = size
    return size


def _image_stim(win, filename, size=None):
    if size is None:
        size = _scaled_size(filename)
    key = (win, filename, size)
    stim = _IMAGE_CACHE.get(key)
    if stim is None:
        pos = (0, IMAGE_Y_OFFSET.get(filename, 0))
        stim = visual.ImageStim(win, image=pool(filename), size=size, pos=pos)
        _IMAGE_CACHE[key] = stim
    return stim


def _text_stim(win, text, height, color, color_space):
    key = (win, text, height, color, color_space)
    stim = _TEXT_CACHE.get(key)
    if stim is None:
        stim = visual.TextStim(win, text=text, color=color, colorSpace=color_space,
                               height=height, wrapWidth=1400, alignText='center',
                               anchorHoriz='center')
        _TEXT_CACHE[key] = stim
    return stim


def preload_stimuli(win):
    """Build and warm every stimulus that appears inside a timed trial.

    Each one is drawn once into the back buffer, which is then cleared without
    a flip -- nothing reaches the screen, but the texture upload and the first
    glyph rasterisation are already paid for by the time trial 1 starts.
    """
    for filename in (FACE_IMAGE, HOUSE_IMAGE):
        _image_stim(win, filename).draw()
    for cue in CUES:
        _text_stim(win, cue, 40, CUE_COLOR, 'rgb255').draw()
    for prompt in (VIVIDNESS_PROMPT, TIME_TO_IMAGINE_PROMPT):
        _text_stim(win, prompt, 30, FG_COLOR, 'rgb').draw()
    win.clearBuffer()


def draw_blank(win):
    return win.flip()   # background color already set on the window


def draw_text(win, text, height=30, color=FG_COLOR, color_space='rgb'):
    _text_stim(win, text, height, color, color_space).draw()
    return win.flip()


def wait_keypress(win, keys=None):
    """Wait for one of `keys`; save and shut down cleanly on Escape."""
    event.clearEvents()
    while True:
        pressed = event.getKeys(keyList=(keys or []) + ['escape'])
        if pressed:
            if pressed[0] == 'escape':
                emergency_quit()
            return pressed[0]
        # Yield the GIL between polls. event.getKeys() pumps the pyglet event
        # queue on every call, so an unthrottled `while True` here saturates
        # the interpreter -- during the rating screens, with recording live --
        # and starves the OpenGaze socket threads. 1 ms keeps keypress latency
        # far below one frame while leaving the CPU almost entirely free.
        time.sleep(0.001)


def draw_image(win, filename, size=None):
    _image_stim(win, filename, size).draw()
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
    "How vivid?\n" 
    "Least (1) - Most (4)"
)

TIME_TO_IMAGINE_PROMPT = (
    "How fast?\n"
    "Slow (1) - Fast (4)"
)


def get_rating(win, prompt_text):
    draw_text(win, prompt_text)
    key = wait_keypress(win, keys=['1', '2', '3', '4'])
    return int(key)


def show_trial_intro(win, tracker, tag, first_image, second_image):
    t0 = draw_image(win, first_image)
    if tracker:
        tracker.log(f"{tag}_TrialIntro_{first_image}_at_{flip_time_ms(t0):.3f}")
    wait_ms(T_INTRO_IMG)

    draw_blank(win)
    wait_ms(T_INTRO_BLANK)

    t0 = draw_image(win, second_image)
    if tracker:
        tracker.log(f"{tag}_TrialIntro_{second_image}_at_{flip_time_ms(t0):.3f}")
    wait_ms(T_INTRO_IMG)


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

    # -- 2. Blank, pre-cue (500 ms) --------------------------------------------
    t0_blank = draw_blank(win)
    if tracker:
        tracker.log(f"{tag}_StartPrecueBlank_at_{flip_time_ms(t0_blank):.3f}")
    wait_ms(T_PRECUE_BLANK)
    if tracker:
        tracker.log(f"{tag}_EndPrecueBlank_at_{libtime.get_time():.3f}")

    # -- 3. H/F cue in center of screen (300 ms) -- imagery only ---------------
    if mode != 'perception':
        t0 = draw_text(win, cue, height=40, color=CUE_COLOR, color_space='rgb255')
        if tracker:
            tracker.log(f"{tag}_StartCue_{cue}_at_{flip_time_ms(t0):.3f}")
        wait_ms(T_CUE)
        if tracker:
            tracker.log(f"{tag}_EndCue_at_{libtime.get_time():.3f}")

    # -- 4. Blank imagery period (imagery) or cued image (perception) ---------
    if mode == 'perception':
        t0 = draw_image(win, cued_image)
        if tracker:
            tracker.log(f"{tag}_StartPerceptionImage_{cued_image}_at_{flip_time_ms(t0):.3f}")
        wait_ms(T_PERCEPTION_IMG)
    else:
        t0 = draw_blank(win)
        if tracker:
            tracker.log(f"{tag}_StartImageryBlank_cued_{cue}_at_{flip_time_ms(t0):.3f}")
        wait_ms(T_IMAGERY_BLANK)
    if tracker:
        tracker.log(f"{tag}_EndStep4_at_{libtime.get_time():.3f}")

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

    # -- ET: log trial variables, THEN stop recording -------------------------
    # log_var() -> tracker.log() -> OpenGaze "SET USER_DATA", which only reaches
    # the gaze file via the sample stream; libopengaze.log() also skips the
    # tracker entirely once self.recording is False. Logging after
    # stop_recording() therefore drops every variable silently, so do it here,
    # while the trailing ITI blank is up and recording is still running.
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

    # -- CSV log (non-training trials only) -----------------------------------
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
    # Training passes a throwaway list and no log file; only register the real
    # one, so an Escape during training cannot clobber the live session state.
    if not is_training:
        register_session(log_rows=log_rows, log_file=log_file,
                         without_tracker=without_tracker)

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
# Escape can be pressed from any screen, but wait_keypress() has no access to
# the tracker or the log. Whatever a clean shutdown needs is registered here as
# soon as it exists, so emergency_quit() can always do the full job.
_SESSION = {
    "win": None, "tracker": None, "disp": None,
    "log_rows": None, "log_file": None, "without_tracker": False,
}


def register_session(**kwargs):
    """Record what an Escape needs in order to shut down cleanly."""
    for key, value in kwargs.items():
        if key not in _SESSION:
            raise KeyError(f"unknown session key: {key}")
        if value is not None:
            _SESSION[key] = value


def quit_and_save(win, tracker, disp, log_rows, log_file, without_tracker):
    """Save whatever has been logged, close everything down, exit.

    tracker.close() is not optional. OpenGaze runs three NON-daemon threads
    whose loops only end on the shutdown signal close() sends, so skipping it
    makes the sys.exit() inside core.quit() block forever joining them --
    leaving the fullscreen window up and the process needing to be killed.
    Each step is guarded so that one failure cannot strand the rest.
    """
    if log_rows is not None and log_file is not None:
        try:
            save_csv(log_rows, log_file, without_tracker)
        except Exception as exc:  # noqa: BLE001 -- shutdown guard, must not skip the remaining steps
            print(f"Warning: could not save the behavioural log: {exc}")

    if tracker:
        try:
            tracker.close()
        except Exception as exc:  # noqa: BLE001 -- shutdown guard, must not skip the remaining steps
            print(f"Warning: tracker.close() failed: {exc}")

    # pygaze's Display wraps pygaze.expdisplay, which *is* `win` -- so
    # disp.close() and win.close() close the same PsychoPy window. Calling both
    # re-enters an unguarded Window.close() and raises, which used to happen
    # before core.quit() and so blocked the exit. Close once.
    try:
        if disp is not None:
            disp.close()
        elif win is not None:
            win.close()
    except Exception as exc:  # noqa: BLE001 -- shutdown guard, must not skip core.quit() below
        print(f"Warning: display close failed: {exc}")

    core.quit()


def emergency_quit():
    """Escape route: shut down using whatever the session has registered."""
    quit_and_save(_SESSION["win"], _SESSION["tracker"], _SESSION["disp"],
                  _SESSION["log_rows"], _SESSION["log_file"],
                  _SESSION["without_tracker"])


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
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # noqa: DTZ005 -- local wall-clock time for a human-readable filename
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
    pygaze_settings.BGC        = (70, 70, 70)  # dark gray

    disp = Display()
    win  = pygaze.expdisplay  # window PyGaze created, stored on the module
    preload_stimuli(win)      # build every trial stimulus now, not mid-trial
    if without_tracker:
        tracker = None
        print("without_tracker=1: running WITHOUT eye tracker, no ET data will be logged.")
    else:
        tracker = EyeTracker(disp, trackertype="opengaze", logfile=et_log)
    register_session(win=win, disp=disp, tracker=tracker,
                     without_tracker=without_tracker)
    return disp, win, tracker
