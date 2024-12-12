from typing import Any, List, Callable
import cv2
import insightface
import threading

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_face # get_one_face, get_many_faces, find_similar_face
# from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'

SWAPPER_MOD = 'inswapper_128.onnx'

def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path(f'../models/{SWAPPER_MOD}')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
    return FACE_SWAPPER

def clear_face_swapper() -> None:
    global FACE_SWAPPER

    FACE_SWAPPER = None

def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, [f'https://huggingface.co/datasets/OwlMaster/gg2/resolve/main/{SWAPPER_MOD}'])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not get_face(cv2.imread(roop.globals.source_path)):
        update_status('No face in source path detected.', NAME)
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    clear_face_swapper()
    # clear_face_reference()



import filetype
import re
import ffmpeg
import numpy as np

def frames_generator(media_path, width, height):
    input_arg = {}
    '''
    if from_ms  : input_arg['ss'] = form_ms / 1000 
    if to_ms    : input_arg['to'] = to_ms / 1000
    '''
    process = (
        ffmpeg
        .input(media_path, **input_arg)
        .output( 'pipe:', format='rawvideo', pix_fmt='bgr24', loglevel="quiet")  # Output raw RGB frames
        .run_async(pipe_stdout=True) # , pipe_stderr=True)
    )
    
    while 1:        
        frame = process.stdout.read(width * height * 3)
        if not frame: break
        
        frame = np.frombuffer(frame, np.uint8).reshape([height, width, 3])
        yield frame
    
    # process.stdin.close()
    process.wait()

def get_frames(media_path):
    
    mime = filetype.guess(media_path).mime
    
    if (mime == 'image/gif') or re.fullmatch(r'video/.*', mime):
          probe   = ffmpeg.probe(media_path)
          info    = next(t for t in probe['streams'] if t['codec_type'] == 'video')

          width, height, nb_frames, codec_name, pix_fmt, frame_rate = (
              info.get(n) for n in ('width', 'height', 'nb_frames', 'codec_name', 'pix_fmt', 'r_frame_rate') 
          )

          width       = int(width or 0)
          height      = int(height or 0)            
          nb_frames   = int(nb_frames or 0)
          frame_rate  = eval(frame_rate) if frame_rate else None
          
          print('# video:', mime, 'url:', media_path, 'fourcc:', codec_name, 'pix:', pix_fmt, 'frames:', nb_frames, 'fps:', frame_rate)
            
          return frames_generator(media_path, width=width, height=height) # frame_rate # , (w, h, 0) # width, height, channel

    elif re.fullmatch(r'image/.*', mime):
        
        if 0: # url in FRAMECACHE:
            # cache
            frames = FRAMECACHE[url]
        else:
            # fetch
            print('# image:', mime, 'url:', media_path)

            if 0: # remote:
                '''
                res = await self.get(url, headers=headers, body='byte')
                img = np.asarray(bytearray(res), dtype="uint8")
                
                frames = (cv2.imdecode(img, cv2.IMREAD_COLOR), 0)  
                '''
            else:      
                frame = cv2.imread(media_path)

            # FRAMECACHE[url] = frames
        
        return [frame]
        
    else:
        assert 0, f'TODO: mime {mime}'


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    reference_face = None # if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
    result = process_frame(source_face, reference_face, target_frame)
    cv2.imwrite(output_path, result)

def process_video(source_path, target_path):
    # roop.processors.frame.core.process_video(source_path, target_path, process_frames)
    process_frames(source_path, target_path)

def process_frames(source_path, target_path): # , update: Callable[[], None]) -> None:
    source = get_frames(source_path)
    target = get_frames(target_path)

    for iface, source_frame in enumerate(source):
        source_face = get_face(source_frame)
        
        first_frame = video_path = video = wi = hi = 0

        for it, target_frame in enumerate(target):
            
            frame = process_frame(source_face, target_frame)  
            
            if it == 0:
                first_frame = frame
                hi, wi, ch = frame.shape # height width channel
                print(f'# dimension: {wi}x{hi} channel: {ch}')

            elif it == 1:
                # prepare video output
                import uuid
                video_path = f'{str(uuid.uuid4())}.mp4'
                video = open_video_output( video_path = video_path, shape = (wi, hi) )
                video.stdin.write(first_frame.tobytes())

            if video:
                video.stdin.write(frame.tobytes())

            if (it % 10) == 0:  
                print(f'# frame {iface}:{it}')

        if video:
            video.stdin.close()
            video.wait()
        break # TODO: next faces

    '''
    source_face = get_face(cv2.imread(source_path), index=roop.globals.face_id)
    reference_face = None # if roop.globals.many_faces else get_face_reference()
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(source_face, reference_face, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()
    '''
def process_frame(source_face, target_frame):
    # temporary rotate
    rc = None
    match roop.globals.rotate:
        case  1: rc = (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)
        case -1: rc = (cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE)
        case  2: rc = (cv2.ROTATE_180, cv2.ROTATE_180)   

    if rc: target_frame = cv2.rotate(target_frame, rc[0])    

    target_face = get_face(target_frame, index=roop.globals.base_id)

    # swap
    if target_face:
        target_frame = get_face_swapper().get(target_frame, target_face, source_face, paste_back=True)
    
    # rotate original
    if rc: target_frame = cv2.rotate(target_frame, rc[1])   

    return target_frame

def open_video_output(video_path, shape, fps=30):    
    video_quality = 1 # (roop.globals.output_video_quality + 1) * 51 // 100

    process = (
      ffmpeg
      .input(
          'pipe:0',  # Input from stdin
          format      = 'rawvideo',
          pix_fmt     = 'bgr24',
          s           = f'{shape[0]}x{shape[1]}', # '1280x720', # f'{wi}x{hi}',
          framerate   = fps
      )
      .output(
          video_path, 
          crf         = video_quality,
          vcodec      = 'libx264',
          pix_fmt     = 'yuv420p',
          preset      = 'veryslow',
          loglevel    = 'quiet'
      )
      .overwrite_output()
      .run_async(pipe_stdin=True) # , quiet=True
    )
    return process


