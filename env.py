from pathlib import Path

ASSETS_PATH = Path(__file__).parent/ 'assets'

FONTS_ROOT = Path(f'{ASSETS_PATH}/font/static')
MANGA109_ROOT = Path('data/Manga109s_released_2021_12_30')
DATA_ROOT = Path('./data')
BACKGROUND_DIR = Path(f'{MANGA109_ROOT}/background')
DATA_SYNTHETIC_ROOT = Path(f'{MANGA109_ROOT}/synthetic')
TRAIN_ROOT = Path('./out')