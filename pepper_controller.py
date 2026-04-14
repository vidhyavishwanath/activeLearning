"""
pepper_controller.py
--------------------
Action dispatcher for Pepper robot.
Supports two input modes (select at startup):
  1. Speech  -- Pepper listens for trigger words via ALSpeechRecognition
  2. Keyboard -- type a key and press Enter

Speech trigger words:
  "next"      -> Turn Passing   (blink + look up)
  "okay"      -> Acknowledgement (says "Okay")
  "yes"       -> Answer Yes      (nod + "Yes")
  "no"        -> Answer No       (shake + "No")
  "query"     -> Make Query      (keyboard prompt for details, then Pepper speaks)
  "positive"  -> Label current config as a positive example of CONCEPT
  "negative"  -> Label current config as a negative example of CONCEPT
  "stop"      -> Exit

Setup:
  1. Press Pepper's chest button -- it will say its IP address
  2. scp this file:  scp pepper_controller.py nao@<ip>:/home/nao/
  3. SSH in:         ssh nao@<ip>   (password: nao)
  4. Run:            python pepper_controller.py
"""

import qi
import time
import sys
import socket
import struct
import json
from scene_understanding import build_scene
from relations import get_relation, ConceptLearner

ROBOT_IP = "128.237.235.109"
PORT     = 9559

# Your Mac's IP on the same network as Pepper.
# Run this on your Mac to find it:  ipconfig getifaddr en0
MAC_IP   = "172.26.37.191"  # <-- replace this
MAC_PORT = 9600

# HeadPitch for watching the table (positive = down on Pepper).
# 0.4 rad ~23 degrees down; tune this to match your table height.
TABLE_PITCH = 0.4

# HeadPitch for the bottom camera to see the table surface during a scan.
# The bottom camera is under the chin and already angles down, so less pitch
# is needed than TABLE_PITCH. Start at 0.1 and tune up if the table is cut off.
SCAN_PITCH = 0.1

# Minimum confidence to accept a recognised word (0.0 - 1.0)
CONFIDENCE_THRESHOLD = 0.4

# ── Learning config ────────────────────────────────────────────────────────────
# Set CONCEPT to whichever object is being taught this session.
# Set LEARNING_MODE to 'supervised' (HOUSE) or 'active' (SNOWMAN).
CONCEPT       = 'house'
LEARNING_MODE = 'active'    # 'supervised' or 'active'

VOCAB = ["next", "okay", "yes", "no", "query", "scan", "positive", "negative", "stop"]


def detect_shapes(session):
    """
    Grab a frame from Pepper's bottom camera, send it to the Mac perception
    server, and return the list of shape dicts.
    """
    # Grab raw frame from Pepper's bottom camera
    video   = session.service("ALVideoDevice")
    name_id = video.subscribeCamera("perc_cam", 1, 2, 11, 10)
    # (name, camera=1=bottom RGB, resolution=2=VGA, colorspace=11=RGB, fps=10)
    time.sleep(0.2)
    result  = video.getImageRemote(name_id)
    video.unsubscribe(name_id)

    if result is None or result[6] is None:
        print("[DETECT] Could not grab frame from camera.")
        return []

    w, h    = result[0], result[1]
    raw     = bytes(result[6])

    # Send to Mac perception server
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect((MAC_IP, MAC_PORT))

        # Send header then raw RGB bytes
        sock.sendall(struct.pack("!II", w, h))
        sock.sendall(raw)

        # Receive JSON response
        length_data = _recv_all(sock, 4)
        length      = struct.unpack("!I", length_data)[0]
        payload     = _recv_all(sock, length)
        sock.close()

        shapes = json.loads(payload.decode("utf-8"))
        return shapes

    except Exception as e:
        print("[DETECT] Error talking to perception server: {}".format(e))
        return []


def _recv_all(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise RuntimeError("Connection closed mid-receive")
        data += chunk
    return data


def connect(ip, port):
    session = qi.Session()
    try:
        session.connect("tcp://{}:{}".format(ip, port))
    except RuntimeError:
        print("Could not connect to Pepper at {}:{}".format(ip, port))
        print("Check ROBOT_IP and that you are on the same network.")
        sys.exit(1)
    return session


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def turn_passing(motion, leds):
    print("[ACTION] Turn Passing")
    motion.angleInterpolation("HeadPitch", [-0.3], [0.5], True)
    for _ in range(3):
        leds.fadeRGB("FaceLeds", 0x0000FF, 0.1)
        time.sleep(0.2)
        leds.fadeRGB("FaceLeds", 0x000000, 0.1)
        time.sleep(0.2)
    leds.fadeRGB("FaceLeds", 0xFFFFFF, 0.3)
    motion.angleInterpolation(
        ["HeadYaw", "HeadPitch"],
        [[0.0], [TABLE_PITCH]],
        [[0.6], [0.6]],
        True
    )


def acknowledgement(tts):
    print("[ACTION] Acknowledgement")
    tts.say("Okay")


def answer_yes(motion, tts):
    print("[ACTION] Answer - Yes")
    tts.say("Yes")
    motion.angleInterpolation(
        "HeadPitch",
        [TABLE_PITCH + 0.3, TABLE_PITCH - 0.1, TABLE_PITCH],
        [0.4, 0.8, 1.1],
        True
    )


def answer_no(motion, tts):
    print("[ACTION] Answer - No")
    tts.say("No")
    motion.angleInterpolation(
        ["HeadYaw",                       "HeadPitch"],
        [[0.4, -0.4, 0.0],                [TABLE_PITCH, TABLE_PITCH, TABLE_PITCH]],
        [[0.4,  0.8, 1.1],                [0.4,         0.8,         1.1]],
        True
    )


def save_debug_frame(session):
    """Save a photo from Pepper's bottom camera using ALPhotoCapture."""
    photo = session.service("ALPhotoCapture")
    photo.setResolution(2)
    photo.setCameraID(1)
    photo.setPictureFormat("jpg")
    result = photo.takePicture("/home/nao/student_projects/s26_mm", "debug_frame")
    print("[DEBUG] Frame saved to: {}".format(result))


def debug_scene(session, tts):
    """Run perception, print and speak the detected relations."""
    motion    = session.service("ALMotion")
    awareness = session.service("ALBasicAwareness")

    awareness.stopAwareness()
    motion.angleInterpolation(
        ["HeadYaw", "HeadPitch"],
        [[0.0], [SCAN_PITCH]],
        [[0.9], [0.9]],
        True
    )
    time.sleep(1.0)
    save_debug_frame(session)

    print("[DEBUG] Scanning scene...")
    shapes = detect_shapes(session)
    scene  = build_scene(shapes, frame_width=640, frame_height=480)

    print("[DEBUG] Shapes detected: {}".format(len(shapes)))
    for s in shapes:
        print("  {} {} at ({}, {})  bbox [{},{}]->[{},{}]".format(
            s['color'], s['label'], s['cx'], s['cy'],
            s['left'], s['top'], s['right'], s['bottom']
        ))

    print("[DEBUG] Relations:")
    for r in scene['relations']:
        print("  {} {} {}".format(
            r['subject']['color'] + ' ' + r['subject']['label'],
            r['relation'],
            r['object']['color']  + ' ' + r['object']['label']
        ))

    print("[DEBUG] Config: {}".format(scene['config']))

    awareness.startAwareness()
    motion.angleInterpolation(
        ["HeadYaw", "HeadPitch"],
        [[0.0], [-0.1]],
        [[0.4], [0.4]],
        True
    )

    if shapes:
        tts.say("I see " + scene['config'])
    else:
        tts.say("I do not see any shapes.")

    motion.angleInterpolation(
        ["HeadYaw", "HeadPitch"],
        [[0.0], [TABLE_PITCH]],
        [[0.4], [0.4]],
        True
    )


def make_query(tts):
    print("[ACTION] Make Query")
    position = ""
    while position not in ("top", "bottom"):
        position = raw_input("  Position (top / bottom): ").strip().lower()
        if position not in ("top", "bottom"):
            print("  Please enter 'top' or 'bottom'.")
    obj = raw_input("  Object name: ").strip()
    if not obj:
        print("  No object name entered, cancelling query.")
        return
    phrase = "Can you replace the {} piece with {}?".format(position, obj)
    print("  Pepper says: \"{}\"".format(phrase))
    tts.say(phrase)


def label_example(is_positive, learner, session, tts):
    """
    Detect the current config, label it, update the learner, speak the
    acknowledgement, and (in active mode) automatically generate and speak
    a query for the most informative next example.
    """
    shapes   = detect_shapes(session)
    relation = get_relation(shapes)

    if relation is None:
        print("[LABEL] No shapes detected.")
        tts.say("I cannot see anything. Please place the blocks in front of me.")
        return

    label_word = "IS" if is_positive else "IS NOT"
    print("[LABEL] {} {} a {}  (step {})".format(
        relation['top'] + ' on ' + relation['bottom'],
        label_word,
        CONCEPT,
        learner.step + 1
    ))

    learner.update(relation, is_positive)
    conf = learner.confidence(relation)
    print("[LABEL] Confidence now: {:.0%}".format(conf))

    # Always acknowledge the label
    acknowledgement(tts)

    # Active mode: auto-generate the most informative query
    if learner.should_query():
        query = learner.generate_query(relation)
        if query:
            swap = query['swap_to'] if query['swap_to'] else "something different"
            phrase = "Can you replace the {} piece with {}?".format(
                query['position'], swap)
            print("[QUERY] {}".format(phrase))
            tts.say(phrase)


def dispatch(word, motion, leds, tts, session, learner):
    """Map a recognised word to its action. Returns True to keep running, False to exit."""
    if word == "next":
        turn_passing(motion, leds)
    elif word == "okay":
        acknowledgement(tts)
    elif word == "yes":
        answer_yes(motion, tts)
    elif word == "no":
        answer_no(motion, tts)
    elif word == "query":
        make_query(tts)
    elif word == "scan":
        debug_scene(session, tts)
    elif word == "positive":
        label_example(True,  learner, session, tts)
    elif word == "negative":
        label_example(False, learner, session, tts)
    elif word == "summary":
        learner.summary()
    elif word == "stop":
        return False
    return True


# ---------------------------------------------------------------------------
# Input loops
# ---------------------------------------------------------------------------

def run_speech_loop(motion, leds, tts, session, learner):
    """Listen for trigger words and dispatch actions."""
    asr    = session.service("ALSpeechRecognition")
    memory = session.service("ALMemory")

    asr.setLanguage("English")
    asr.pause(True)
    asr.setVocabulary(VOCAB, False)
    asr.pause(False)
    asr.subscribe("pepper_controller")
    print("Listening for: {}".format(", ".join(VOCAB)))
    print("Say 'stop' to exit.\n")

    memory.insertData("WordRecognized", ["", 0.0])

    try:
        while True:
            time.sleep(0.1)
            result = memory.getData("WordRecognized")
            if not result or len(result) < 2:
                continue
            word, confidence = result[0], result[1]
            if not word or confidence < CONFIDENCE_THRESHOLD:
                continue

            memory.insertData("WordRecognized", ["", 0.0])
            print("Heard: '{}' (confidence {:.2f})".format(word, confidence))

            asr.unsubscribe("pepper_controller")
            tts.say("mm hm")

            keep_going = dispatch(word, motion, leds, tts, session, learner)
            if not keep_going:
                break
            asr.subscribe("pepper_controller")
            memory.insertData("WordRecognized", ["", 0.0])

    except KeyboardInterrupt:
        pass
    finally:
        try:
            asr.unsubscribe("pepper_controller")
        except Exception:
            pass


MENU = """
--- Pepper Controller  [concept: {concept}  mode: {mode}  steps: {steps}  conf: {conf}] ---
  t  Turn Passing       (blink + look up)
  a  Acknowledgement    ("Okay")
  y  Answer Yes         (nod + "Yes")
  n  Answer No          (shake + "No")
  q  Make Query         (prompted speech)
  d  Debug Scan         (detect shapes + speak relations)
  p  Positive label     ("this IS a {concept}")
  b  Negative label     ("this is NOT a {concept}")
  s  Summary            (print what Pepper has learned)
  x  Exit
> """

KEY_MAP = {
    "t": "next",
    "a": "okay",
    "y": "yes",
    "n": "no",
    "q": "query",
    "d": "scan",
    "p": "positive",
    "b": "negative",
    "s": "summary",
    "x": "stop",
}


def run_keyboard_loop(motion, leds, tts, session, learner):
    """Dispatch actions from keyboard input."""
    while True:
        conf   = learner.confidence(None) if learner.step == 0 else 0.0
        prompt = MENU.format(
            concept=CONCEPT,
            mode=LEARNING_MODE,
            steps=learner.step,
            conf="{:.0%}".format(conf),
        )
        try:
            key = raw_input(prompt).strip().lower()
        except (KeyboardInterrupt, EOFError):
            break
        word = KEY_MAP.get(key)
        if word is None:
            print("Unknown key '{}'. Use t/a/y/n/q/d/p/b/s/x.".format(key))
            continue
        if not dispatch(word, motion, leds, tts, session, learner):
            break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Connecting to Pepper at {}:{}...".format(ROBOT_IP, PORT))
    session = connect(ROBOT_IP, PORT)
    print("Connected.")

    motion  = session.service("ALMotion")
    posture = session.service("ALRobotPosture")
    tts     = session.service("ALTextToSpeech")
    leds    = session.service("ALLeds")

    motion.wakeUp()
    posture.goToPosture("StandInit", 0.5)
    motion.angleInterpolation(
        ["HeadYaw", "HeadPitch"],
        [[0.0], [TABLE_PITCH]],
        [[0.5], [0.5]],
        True
    )

    learner = ConceptLearner(mode=LEARNING_MODE)
    print("Learning mode: {}  |  Concept: {}".format(LEARNING_MODE, CONCEPT))

    mode = raw_input("Input mode: (s)peech or (k)eyboard? [s] ").strip().lower()
    if mode == "k":
        run_keyboard_loop(motion, leds, tts, session, learner)
    else:
        run_speech_loop(motion, leds, tts, session, learner)

    print("\n--- Final session summary ---")
    learner.summary()

    print("Returning to rest posture...")
    posture.goToPosture("StandInit", 0.5)
    motion.rest()
    print("Done.")


if __name__ == "__main__":
    main()
