import cv2
import os
from tqdm import tqdm
import pickle
from glob import glob
import sys

ROOT = os.getcwd()
BASE = "original"
TARGET = "data"

SUB = "Subj_0"
PERSON_NUM = range(3, 13)
ACTIONS = ['_deep', '_reading', '_stationary', '_talking', '_watching']


if not os.path.exists(BASE):
    sys.exit("No original directory!")

if not os.path.exists(TARGET):
    os.makedirs(TARGET)

for number in PERSON_NUM:
    sub_path = os.path.join(SUB + '{0:02d}'.format(number))
    BASE_path = os.path.join(ROOT, BASE, sub_path)
    TARGET_path = os.path.join(ROOT, TARGET, sub_path)

    if not os.path.exists(TARGET_path):
        os.makedirs(TARGET_path)

    for action in ACTIONS:
        base_action_dir = os.path.join(BASE_path, sub_path + action)
        target_action_dir = os.path.join(TARGET_path, sub_path + action)

        if not os.path.exists(target_action_dir):
            os.makedirs(target_action_dir)

        base_images_path = os.path.join(base_action_dir, "png")
        base_pulse_path  = os.path.join(base_action_dir, 'PulseOX')

        target_images_path = os.path.join(target_action_dir, "image")
        if not os.path.exists(target_images_path):
            os.makedirs(target_images_path)
        target_pulse_path  = target_action_dir

        base_pulse_file = open(os.path.join(base_pulse_path, 'px1_full.pkl'), 'rb')
        target_pulse_file_path = os.path.join(target_pulse_path, "pulse.pkl")
        target_pulse_file = open(target_pulse_file_path, 'wb')

        base_image_path = sorted(glob(os.path.join(base_images_path, '*.png')))
        base_pulse = pickle.load(base_pulse_file, encoding='latin1')['pulseOxRecord']

       
        print(sub_path + action)
        for number, base_image in tqdm(enumerate(base_image_path), total = len(base_image_path), desc = 'Images'):
            if number % 2 == 0:
                file_name = os.path.split(base_image)[-1]
                target_image = os.path.join(target_images_path, file_name)
                img = cv2.imread(base_image, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
                cv2.imwrite(target_image, img)
            else:
                continue

        target_pulse = []
        for number, base_pulse in tqdm(enumerate(base_pulse), total = len(base_pulse), desc = 'Pulses'):
            if number % 4 == 0:
                target_pulse.append(base_pulse)
        pickle.dump(target_pulse, target_pulse_file)

        base_pulse_file.close()
        target_pulse_file.close()