from pathlib import Path

ASSETS_PATH = Path(__file__).parent/ 'assets'

FONTS_ROOT = Path(f'{ASSETS_PATH}/font/static')
MANGA109_ROOT = Path('../Manga109s_released_2021_12_30')
BACKGROUND_DIR = Path(f'{MANGA109_ROOT}/background')
DATA_SYNTHETIC_ROOT = Path(f'{MANGA109_ROOT}/synthetic')
TRAIN_ROOT = Path('./out')
