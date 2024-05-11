import traceback
from pathlib import Path

import cv2
import fire
import pandas as pd
from tqdm.contrib.concurrent import thread_map
from tqdm import tqdm
from hide_warnings import hide_warnings
# from manga_ocr_dev.env import FONTS_ROOT, DATA_SYNTHETIC_ROOT
from generator import SyntheticDataGenerator

DATA_SYNTHETIC_ROOT = Path('/teamspace/studios/this_studio/manga-ocr/manga_ocr_dev/data/manga/synthetic')
FONTS_ROOT = Path('/teamspace/studios/this_studio/manga-ocr/assets/font/static')

generator = SyntheticDataGenerator()

@hide_warnings()
def f(args):
    try:
        i, source, id_, text = args
        filename = f'{id_}.jpg'
        img, text_gt, params = generator.process(text)

        cv2.imwrite(str(OUT_DIR / filename), img)

        font_path = Path(params['font_path']).relative_to(FONTS_ROOT)
        ret = source, id_, text_gt, params['vertical'], str(font_path)
        
        return ret

    except Exception as e:
        pass


def run(begin=0, end=1, n_random=1000, n_limit=None, max_workers=16):
    """
    :param package: number of data package to generate
    :param n_random: how many samples with random text to generate
    :param n_limit: limit number of generated samples (for debugging)
    :param max_workers: max number of workers
    """
    for package in tqdm(range(begin,end)):
        package = f'{package:04d}'
        lines = pd.read_csv(DATA_SYNTHETIC_ROOT / f'lines/{package}.csv',on_bad_lines='skip',header=None)
        random_lines = pd.DataFrame({
            'source': 'cc-100',
            'id': [f'cc-100_{package}_{i}' for i in range(n_random)]
        })
        lines = pd.concat([random_lines, lines], ignore_index=True, axis=1)
        if n_limit:
            lines = lines.head(n_limit)
        args = [(i, *values) for i, values in enumerate(lines.values)]
        
        global OUT_DIR
        OUT_DIR = DATA_SYNTHETIC_ROOT / 'img' / package
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        data = thread_map(f, args, max_workers=max_workers, desc=f'Processing package {package}')

        data = pd.DataFrame(data, columns=['source', 'id', 'text', 'vertical', 'font_path'])
        meta_path = DATA_SYNTHETIC_ROOT / f'meta/{package}.csv'
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(meta_path, index=False)
        
if __name__ == '__main__':
    fire.Fire(run)