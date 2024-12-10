import threading
from typing import Any, Optional, List
import insightface
import numpy

import roop.globals
from roop.typing import Frame, Face

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=roop.globals.execution_providers)
            FACE_ANALYSER.prepare(ctx_id=0)
    return FACE_ANALYSER


def clear_face_analyser() -> Any:
    global FACE_ANALYSER

    FACE_ANALYSER = None

def get_index(index, count):    
    r = None
    if (index > 0):
        if (index <= count):
            r = index-1
    elif (index < 0):
        if (abs(index) <= count):
            r = index
    return r

def get_face(frame, index=1):
    faces = get_face_analyser().get(frame)
    faces.sort(key = lambda x: (x.bbox[1],x.bbox[0]) if roop.globals.sort == 0 else (x.bbox[0],x.bbox[1]))
    if not faces: return

    _i = get_index(index, len(faces))
    return faces[_i]
