import zipfile

import numpy as np
from cnocr import CnOcr
from cnstd import CnStd
from cnocr.consts import MODEL_VERSION as ocr_version
from cnstd.consts import MODEL_VERSION as std_version
import platform
import shutil
import os
import cv2

OCR_MODEL_NAME = 'densenet-lite-fc'
STD_MODEL_NAME = 'mobilenetv3'


def data_dir_default(mode='cnocr'):
    """
    :return: default data directory depending on the platform and environment variables
    """
    system = platform.system()
    if system == 'Windows':
        return os.path.join(os.environ.get('APPDATA'), mode)
    else:
        return os.path.join(os.path.expanduser("~"), mode)


def data_dir(mode='cnocr'):
    """
    :return: data directory in the filesystem for storage, for example when downloading models
    """
    return os.getenv(mode.upper() + 'HOME', data_dir_default(mode=mode))


def start_engine(model_path='models'):
    if OCR_MODEL_NAME + '.zip' in os.listdir(model_path) and STD_MODEL_NAME + '.zip' in os.listdir(model_path):
        print('初始化模型中...')

        ocr_root = os.path.join(data_dir(mode='cnocr'), ocr_version)
        ocr_root = os.path.join(ocr_root, OCR_MODEL_NAME)
        os.makedirs(ocr_root, exist_ok=True)
        ocr_zip_file_path = os.path.join(ocr_root, OCR_MODEL_NAME + '.zip')
        shutil.copyfile(os.path.join('models', OCR_MODEL_NAME + '.zip'), ocr_zip_file_path)
        with zipfile.ZipFile(ocr_zip_file_path) as zf:
            zf.extractall(os.path.dirname(ocr_root))
        os.remove(ocr_zip_file_path)

        std_root = os.path.join(data_dir(mode='cnstd'), std_version)
        std_root = os.path.join(std_root, STD_MODEL_NAME)
        os.makedirs(std_root, exist_ok=True)
        std_zip_file_path = os.path.join(std_root, STD_MODEL_NAME + '.zip')
        shutil.copyfile(os.path.join('models', STD_MODEL_NAME + '.zip'), std_zip_file_path)
        with zipfile.ZipFile(std_zip_file_path) as zf:
            zf.extractall(os.path.dirname(std_root))
        os.remove(std_zip_file_path)

        ocr_core = CnOcr()
        std_core = CnStd()

        print('初始化完成')
        return ocr_core, std_core
    else:
        print('初始化失败，未找到预加载模型')
        return None


def has_chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


if __name__ == '__main__':
    ocr = CnOcr()
    std = CnStd()
    path = '/images/life.jpg'
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)
    # res = ocr.ocr(img)

    box_info_list = std.detect(img)

    res = ''
    for box_info in box_info_list:
        cropped_img = box_info['cropped_img']
        ocr_res = ocr.ocr_for_single_line(cropped_img)
        res += ''.join(ocr_res) + '\n'
        # print('ocr result: %s' % ''.join(ocr_res))

    print("Predicted Chars:", res)


