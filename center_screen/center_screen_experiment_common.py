import copy
import csv
import ctypes
import os
import socket
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
from psychopy import core, event, visual
from pygaze import libtime
from pygaze._eyetracker.opengaze import OpenGazeTracker as _OpenGazeSocketTracker
from pygaze.display import Display
from pygaze.eyetracker import EyeTracker


# -----------------------------------------------------------------------------
# PATCH: pygaze's OpenGaze socket-reader thread can crash mid-session
# -----------------------------------------------------------------------------
# OpenGazeTracker._process_incoming() (pygaze/_eyetracker/opengaze.py) reads
# the tracker's TCP stream and splits it into messages. If a chunk happens to
# contain no complete message (e.g. it's just a stray trailing "\r\n"), the
# resulting `messages` list is empty, and the very next lines index into it
# unconditionally -- an unhandled IndexError that silently kills this
# background thread. Once it's dead, no more ACKs are ever recorded, so every
# later tracker.log()/log_var()/start_recording()/stop_recording() call (each
# of which blocks waiting for an ACK, retrying for up to 9s) stalls for
# several seconds, on every single screen, for the rest of the session. This
# is what caused the multi-second freezes seen starting partway through the
# perception block for two different participants -- it's a race that gets
# more likely to trigger the longer the session runs, not something specific
# to those trials. Patched here (rather than editing site-packages) because
# pygaze is reinstalled fresh from GitHub on every `uv sync`.
def _process_incoming_patched(self):
    self._debug_print("Incoming Thread started.")

    while self._connected.is_set():
        timeout = False
        with self._socklock:
            try:
                instring = self._sock.recv(self._maxrecvsize)
            except socket.timeout:
                timeout = True
            else:
                instring = instring.decode("utf-8")
        t = time.time()

        if timeout:
            self._debug_print("socket recv timeout")
            continue

        self._debug_print(r"Raw instring: {}".format(instring))

        messages = [msg for msg in instring.splitlines() if msg.strip()]

        # PATCH: nothing usable in this chunk -- leave any unfinished
        # fragment buffered for next time instead of indexing an empty list.
        if not messages:
            time.sleep(0.005)
            continue

        if self._unfinished:
            messages[0] = copy.copy(self._unfinished) + messages[0]
            self._unfinished = ''
        if not messages[-1][-2:] == '/>':
            self._unfinished = messages.pop(-1)

        for msg in messages:
            self._debug_print(r"Incoming: {}".format(msg))
            # PATCH: a message that's non-empty but still malformed (the same
            # "frequently sends malformed XML" behaviour noted above) makes
            # lxml raise inside _parse_msg. Uncaught, that's the same
            # thread-killing failure as the empty-messages case -- drop just
            # this one message instead.
            try:
                command, msgdict = self._parse_msg(msg)
            except Exception:
                self._debug_print(r"Unparseable message, dropped: {}".format(msg))
                continue
            if command == 'ACK':
                self._acklock.acquire()
                self._acknowledgements[msgdict['ID']] = copy.copy(t)
                self._acklock.release()
            self._inlock.acquire()
            if command not in self._incoming.keys():
                self._incoming[command] = {}
            if 'ID' not in msgdict.keys():
                msgdict['ID'] = 'NO_ID'
            if msgdict['ID'] not in self._incoming[command].keys():
                self._incoming[command][msgdict['ID']] = {}
            self._incoming[command][msgdict['ID']]['t'] = copy.copy(t)
            for par, val in msgdict.items():
                self._incoming[command][msgdict['ID']][par] = copy.copy(val)
            if command == 'REC' and self._logging.is_set():
                self._logqueue.put(copy.deepcopy(
                    self._incoming[command][msgdict['ID']]))
            self._inlock.release()
        time.sleep(0.005)
    self._debug_print("Incoming Thread ended.")
    return


_OpenGazeSocketTracker._process_incoming = _process_incoming_patched


SCREEN_W    = 1920
SCREEN_H    = 1080
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

# Timing (ms)
T_ITI            = 1000   # inter-trial interval, at both start and end of each trial
T_PRECUE_BLANK   = 250    # blank between the trial-intro preview and the H/F cue / fixation cross
T_CUE            = 300    # H/F cue (imagery mode) or fixation cross (perception mode), center of screen
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
    for stim in _cross_stim(win):
        stim.draw()
    win.clearBuffer()


_CROSS_CACHE = {}


def _cross_stim(win):
    stims = _CROSS_CACHE.get(win)
    if stims is None:
        v = visual.Line(win, start=(0, -16), end=(0, 16), color=FG_COLOR, colorSpace='rgb', lineWidth=1)
        h = visual.Line(win, start=(-16, 0), end=(16, 0), color=FG_COLOR, colorSpace='rgb', lineWidth=1)
        stims = (v, h)
        _CROSS_CACHE[win] = stims
    return stims


def draw_cross(win):
    for stim in _cross_stim(win):
        stim.draw()
    return win.flip()


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
    draw_image(win, first_image)
    if tracker:
        tracker.log(f"{tag}_TrialIntro_{first_image}_at_{libtime.get_time():.3f}")
    wait_ms(T_INTRO_IMG)

    draw_blank(win)
    wait_ms(T_INTRO_BLANK)

    draw_image(win, second_image)
    if tracker:
        tracker.log(f"{tag}_TrialIntro_{second_image}_at_{libtime.get_time():.3f}")
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

    # -- 2. Blank, pre-cue (500 ms) --------------------------------------------
    draw_blank(win)
    if tracker:
        tracker.log(f"{tag}_StartPrecueBlank_at_{libtime.get_time():.3f}")
    wait_ms(T_PRECUE_BLANK)
    if tracker:
        tracker.log(f"{tag}_EndPrecueBlank_at_{libtime.get_time():.3f}")

    # -- 3. H/F cue (imagery) or fixation cross (perception) -- both 300 ms ----
    if mode != 'perception':
        draw_text(win, cue, height=40, color=CUE_COLOR, color_space='rgb255')
        if tracker:
            tracker.log(f"{tag}_StartCue_{cue}_at_{libtime.get_time():.3f}")
        wait_ms(T_CUE)
        if tracker:
            tracker.log(f"{tag}_EndCue_at_{libtime.get_time():.3f}")
    else:
        draw_cross(win)
        if tracker:
            tracker.log(f"{tag}_StartFixation_at_{libtime.get_time():.3f}")
        wait_ms(T_CUE)
        if tracker:
            tracker.log(f"{tag}_EndFixation_at_{libtime.get_time():.3f}")

    # -- 4. Blank imagery period (imagery) or cued image (perception) ---------
    if mode == 'perception':
        draw_image(win, cued_image)
        if tracker:
            tracker.log(f"{tag}_StartPerceptionImage_{cued_image}_at_{libtime.get_time():.3f}")
        wait_ms(T_PERCEPTION_IMG)
    else:
        draw_blank(win)
        if tracker:
            tracker.log(f"{tag}_StartImageryBlank_cued_{cue}_at_{libtime.get_time():.3f}")
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

    # -- ET: log trial variables ------------------------------------------------
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
            tracker.stop_recording()
        except Exception as exc:  # noqa: BLE001 -- shutdown guard, must not skip the remaining steps
            print(f"Warning: tracker.stop_recording() failed: {exc}")
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
# SAMPLE RATE CHECK
# -----------------------------------------------------------------------------
# pygaze's OpenGaze wrapper hardcodes self.samplerate = 60.0 and never
# verifies it against the connected hardware (see libopengaze.py's own
# "TODO: Compute after streaming some samples?"), so a 150Hz GP3 HD tracker
# would be silently mislabeled. Measure it for real instead: watch the
# device's own CNT counter (already streamed -- enable_send_counter(True) is
# on by default) over a short recording window, and derive Hz from elapsed
# wall time.
KNOWN_SAMPLE_RATES_HZ   = (60, 150)   # Gazepoint GP3 / GP3 HD nominal rates
EXPECTED_SAMPLE_RATE_HZ = 150         # this study's tracker (GP3 HD) -- flagged if unmet


def _latest_cnt(tracker):
    """Sample counter off the most recent REC message, or None if no gaze
    sample has arrived yet."""
    rec = tracker.opengaze._incoming.get('REC', {}).get('NO_ID', {})
    cnt = rec.get('CNT')
    return int(cnt) if cnt is not None else None


def measure_sample_rate(tracker, duration_ms=1000):
    """Watches the sample stream briefly and returns the empirical sampling
    rate in Hz, or None if no samples arrived (e.g. tracker not actually
    connected). Assumes the tracker is already recording."""
    tracker.log("SampleRateCheck_Start")

    # Wait for the first sample so _incoming['REC'] is populated.
    start_cnt = None
    deadline = time.time() + 2.0
    while start_cnt is None and time.time() < deadline:
        start_cnt = _latest_cnt(tracker)
        if start_cnt is None:
            time.sleep(0.005)

    if start_cnt is None:
        tracker.log("SampleRateCheck_NoSamples")
        return None

    t_start = time.time()
    wait_ms(duration_ms)
    t_end = time.time()
    end_cnt = _latest_cnt(tracker)

    tracker.log("SampleRateCheck_End")

    if end_cnt is None or end_cnt <= start_cnt:
        return None
    return (end_cnt - start_cnt) / (t_end - t_start)


def report_sample_rate(win, tracker):
    """Measures and displays the tracker's sampling rate before the
    experiment starts, so a rate mismatch is caught before running a
    participant, not discovered later while analysing the gaze data.

    A rate other than EXPECTED_SAMPLE_RATE_HZ requires the experimenter to
    type Y rather than just press SPACE, so a wrong device (e.g. a 60Hz GP3
    swapped in for the study's 150Hz GP3 HD) can't be waved through by an
    reflexive keypress.
    """
    draw_text(win, "Checking eye tracker sampling rate...")
    hz = measure_sample_rate(tracker)

    if hz is None:
        matches_expected = False
        msg = "Could not detect eye tracker sampling rate\n(no gaze samples received)."
    else:
        nominal = min(KNOWN_SAMPLE_RATES_HZ, key=lambda r: abs(r - hz))
        within_tolerance = abs(hz - nominal) / nominal <= 0.15
        matches_expected = within_tolerance and nominal == EXPECTED_SAMPLE_RATE_HZ
        if within_tolerance:
            msg = f"Eye tracker sampling rate: ~{hz:.0f} Hz ({nominal} Hz)"
        else:
            msg = f"Eye tracker sampling rate: ~{hz:.0f} Hz\n(does not match a known 60/150 Hz rate)"

    print(msg.replace("\n", " "))

    if matches_expected:
        draw_text(win, msg + "\n\nPress SPACE to continue.")
        wait_keypress(win, keys=['space'])
    else:
        print(f"WARNING: expected {EXPECTED_SAMPLE_RATE_HZ} Hz.")
        warning = (f"WARNING: expected {EXPECTED_SAMPLE_RATE_HZ} Hz.\n\n{msg}"
                   "\n\nType Y to continue anyway, or Escape to quit.")
        draw_text(win, warning, color="red")
        wait_keypress(win, keys=['y'])


# -----------------------------------------------------------------------------
# GUARD: Alt (and other Windows "system" keys) can hang a fullscreen pyglet
# window that has no menu bar
# -----------------------------------------------------------------------------
# pyglet's own win32 backend (pyglet/window/win32/__init__.py, _event_syscommand)
# carries the comment "check for ALT key to prevent app from hanging because
# there is no windows menu bar" -- Alt is a Windows *system* key (WM_SYSKEYDOWN/
# WM_SYSKEYUP), and on a borderless/fullscreen window with no menu, Windows'
# default handling for it (DefWindowProc) is exactly the kind of thing that
# can wedge a fullscreen exclusive OpenGL window. pyglet's own mitigation only
# covers one specific WM_SYSCOMMAND case. Requesting exclusive keyboard
# capture is the robust fix: it stops WM_SYSKEYDOWN/UP from ever reaching
# DefWindowProc at all, so Alt (or F10, or the Windows key) can't trigger any
# OS-level handling while this window has focus. Space, 1-4, Escape, etc. are
# all ordinary (non-system) keys and are unaffected.
# Alt sits directly next to Space on a standard keyboard -- a participant
# reaching for "press Space to continue" at a break screen and catching Alt
# instead is a very plausible way to trigger this.
def _enable_exclusive_keyboard(win):
    handle = getattr(win, "winHandle", None)
    set_exclusive = getattr(handle, "set_exclusive_keyboard", None)
    if set_exclusive is None:
        return  # not the pyglet/win32 backend -- nothing to do
    try:
        set_exclusive(True)
    except Exception as exc:  # noqa: BLE001 -- best-effort hardening, must not block startup
        print(f"Warning: could not enable exclusive keyboard capture: {exc}")


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
    _enable_exclusive_keyboard(win)
    preload_stimuli(win)      # build every trial stimulus now, not mid-trial
    if without_tracker:
        tracker = None
        print("without_tracker=1: running WITHOUT eye tracker, no ET data will be logged.")
    else:
        tracker = EyeTracker(disp, trackertype="opengaze", logfile=et_log)
        tracker.start_recording()  # continuous for the whole session; stopped in quit_and_save
        report_sample_rate(win, tracker)
    register_session(win=win, disp=disp, tracker=tracker,
                     without_tracker=without_tracker)
    return disp, win, tracker
