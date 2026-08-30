import copy
import csv
import ctypes
import os
import socket
import time
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

os.environ["DISPTYPE"] = "psychopy"
os.environ["TRACKERTYPE"] = "opengaze"

import pygaze
import pygaze.settings as pygaze_settings
from psychopy import core, event, visual
from pygaze import libtime
from pygaze._eyetracker.opengaze import OpenGazeTracker as _OpenGazeSocketTracker
from pygaze.display import Display
from pygaze.eyetracker import EyeTracker


# PATCH: pygaze's OpenGaze incoming-message thread raises IndexError (and
# silently dies) on a chunk with no complete message, killing ACK handling
# and stalling every later tracker.log()/log_var() call for up to 9s. Patched
# here instead of site-packages since pygaze reinstalls fresh on `uv sync`.
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

        # Nothing usable yet; keep buffering instead of indexing an empty list.
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
            # Malformed XML (GazePoint does this) would otherwise kill this
            # thread the same way -- drop just this message instead.
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
CUES        = ("F", "H")

# Per-image vertical nudge (px) since face/house differ in visual center of mass.
IMAGE_Y_OFFSET = {
    FACE_IMAGE:  -20,
    HOUSE_IMAGE:  20,
}

# Timing (ms)
T_ITI            = 1000   # inter-trial interval, start and end of each trial
T_PRECUE_BLANK   = 250    # blank between the preview and the cue/fixation cross
T_CUE            = 400    # H/F cue (imagery) or fixation cross (perception)
T_IMAGERY_BLANK  = 4000   # blank imagery period (imagery mode)
T_PERCEPTION_IMG = 4000   # cued image display duration (perception mode)
T_INTRO_IMG      = 1500   # each two-stimulus preview image on screen
T_INTRO_BLANK    = 1000   # blank between the two preview images
T_TAG_GAP        = 30     # lets EndStep4 land in a gaze sample (150Hz) before
                          # being overwritten by StartVividnessRating


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
    # hogCPUperiod=0.2: hogging the whole wait starves the OpenGaze socket
    # threads (and thus gaze samples) by holding the GIL.
    core.wait(ms / 1000.0, hogCPUperiod=0.2)


# Stimulus cache: building a PsychoPy stimulus (texture upload, glyph raster)
# inside a trial would jitter stimulus onset, so each is built once and cached.
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
    """Draws every trial stimulus once (buffer cleared, no flip) so texture
    upload/rasterisation is paid for before trial 1 starts."""
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
        # Throttle -- an unthrottled loop here starves the OpenGaze socket threads.
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

    # 2. Pre-cue blank
    draw_blank(win)
    if tracker:
        tracker.log(f"{tag}_StartPrecueBlank_at_{libtime.get_time():.3f}")
    wait_ms(T_PRECUE_BLANK)
    if tracker:
        tracker.log(f"{tag}_EndPrecueBlank_at_{libtime.get_time():.3f}")

    # 3. H/F cue (imagery) or fixation cross (perception)
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

    # 4. Blank imagery period (imagery) or cued image (perception)
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
    # Only register real sessions -- Escape during training must not clobber live state.
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


def run_training(win, training_trials, mode='imagery'):
    run_trials(win, tracker=None, trials=training_trials, log_rows=[],
               start_trial_num=1, is_training=True, mode=mode)
    draw_blank(win)
    event.clearEvents()
    core.wait(0.5)


# EXPERIMENTER QUIT: wait_keypress() has no access to the tracker/log, so
# whatever a clean shutdown needs is registered here as soon as it exists.
_SESSION = {
    "win": None, "tracker": None, "disp": None,
    "log_rows": None, "log_file": None, "without_tracker": False,
}


def register_session(**kwargs):
    """Records what an Escape needs to shut down cleanly."""
    for key, value in kwargs.items():
        if key not in _SESSION:
            raise KeyError(f"unknown session key: {key}")
        if value is not None:
            _SESSION[key] = value


def quit_and_save(win, tracker, disp, log_rows, log_file, without_tracker):
    """Save logged data, close everything down, exit.

    tracker.close() is required -- OpenGaze's three non-daemon threads only
    stop on that signal, so skipping it hangs core.quit() forever. Each step
    is guarded so one failure can't strand the rest.
    """
    if log_rows is not None and log_file is not None:
        try:
            save_csv(log_rows, log_file, without_tracker)
        except Exception as exc:  # noqa: BLE001 -- must not abort shutdown
            print(f"Warning: could not save the behavioural log: {exc}")

    if tracker:
        try:
            tracker.stop_recording()
        except Exception as exc:  # noqa: BLE001 -- must not abort shutdown
            print(f"Warning: tracker.stop_recording() failed: {exc}")
        try:
            tracker.close()
        except Exception as exc:  # noqa: BLE001 -- must not abort shutdown
            print(f"Warning: tracker.close() failed: {exc}")

    # disp.close() and win.close() close the same window -- call only one, or
    # the second (unguarded) Window.close() raises and blocks exit.
    try:
        if disp is not None:
            disp.close()
        elif win is not None:
            win.close()
    except Exception as exc:  # noqa: BLE001 -- must not skip core.quit()
        print(f"Warning: display close failed: {exc}")

    core.quit()


def emergency_quit():
    """Escape route: shut down using whatever the session has registered."""
    quit_and_save(_SESSION["win"], _SESSION["tracker"], _SESSION["disp"],
                  _SESSION["log_rows"], _SESSION["log_file"],
                  _SESSION["without_tracker"])


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
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # noqa: DTZ005 -- local time for filename
    log_file = os.path.join(results_dir, f"log_{task_name}_subj_{subject_nr}_{timestamp}.csv")
    et_log   = os.path.join(results_dir, f"gazepoint_data_{task_name}_subj_{subject_nr}_{timestamp}")
    return log_file, et_log


# pygaze hardcodes samplerate=60.0 and never checks the hardware, which would
# silently mislabel a 150Hz GP3 HD. Measure it for real via the CNT counter.
KNOWN_SAMPLE_RATES_HZ   = (60, 150)   # Gazepoint GP3 / GP3 HD nominal rates
EXPECTED_SAMPLE_RATE_HZ = 150         # this study's tracker (GP3 HD) -- flagged if unmet


def _latest_cnt(tracker):
    """Latest sample counter, or None if nothing has arrived yet."""
    rec = tracker.opengaze._incoming.get('REC', {}).get('NO_ID', {})
    cnt = rec.get('CNT')
    return int(cnt) if cnt is not None else None


def measure_sample_rate(tracker, duration_ms=1000):
    """Empirical sampling rate in Hz, or None if no samples arrive."""
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
    """Checks sampling rate before the session starts. A mismatch requires
    typing Y (not just Space) so a wrong device can't be waved through."""
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


# GUARD: Alt is a Windows "system" key that can hang a fullscreen pyglet
# window with no menu bar (WM_SYSKEYDOWN -> DefWindowProc). Exclusive keyboard
# capture stops it from ever reaching DefWindowProc. Worth guarding since Alt
# sits right next to Space, which participants are told to press.
def _enable_exclusive_keyboard(win):
    handle = getattr(win, "winHandle", None)
    set_exclusive = getattr(handle, "set_exclusive_keyboard", None)
    if set_exclusive is None:
        return  # not the pyglet/win32 backend -- nothing to do
    try:
        set_exclusive(True)
    except Exception as exc:  # noqa: BLE001 -- best-effort, must not block startup
        print(f"Warning: could not enable exclusive keyboard capture: {exc}")


def setup_display_and_tracker(without_tracker, et_log):
    # Let PyGaze own the window (prevents double-window on startup).
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
