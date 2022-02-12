import requests
from enum import Enum
import os
from tqdm import tqdm
from appdirs import user_cache_dir
from pathlib import Path

MODEL_4_DND = "https://raw.githubusercontent.com/MilaNLProc/fairface/master/models/res34_fair_align_multi_4_20190809.pt"
MODEL_7_DND = "https://raw.githubusercontent.com/MilaNLProc/fairface/master/models/res34_fair_align_multi_7_20190809.pt"
MMOD_DND = "https://raw.githubusercontent.com/MilaNLProc/fairface/master/models/mmod_human_face_detector.dat"
SHAPE_DND = "https://raw.githubusercontent.com/MilaNLProc/fairface/master/models/shape_predictor_5_face_landmarks.dat"

MODEL_4 = "res34_fair_align_multi_4_20190809.pt"
MODEL_7 = "res34_fair_align_multi_7_20190809.pt"
MMOD = "mmod_human_face_detector.dat"
SHAPE = "shape_predictor_5_face_landmarks.dat"

def download_all_models():

    to_down = zip([MODEL_4, MODEL_7, MMOD, SHAPE], [MODEL_4_DND, MODEL_7_DND, MMOD_DND, SHAPE_DND])

    for name, link in to_down:
        cache_dir = get_cache_directory("models")
        filepath = os.path.join(cache_dir, name)
        if not os.path.exists(filepath):
            download_with_progress(link, filepath)

def get_cache_directory(elements="image"):
    """
    Returns the cache directory on the system
    """
    appname = f"fairface-{elements}"
    appauthor = "demographer"
    cache_dir = user_cache_dir(appname, appauthor)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    return cache_dir


def download_with_progress(url, destination):
    """
    Downloads a file with a progress bar
    :param url: url from which to download from
    :destination: file path for saving data
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)
    print(destination)
    with tqdm.wrapattr(open(destination, "wb"), "write",
                       miniters=1, desc=url.split('/')[-1],
                       total=int(response.headers.get('content-length', 0))) as fout:
        for chunk in response.iter_content(chunk_size=4096):
            fout.write(chunk)

