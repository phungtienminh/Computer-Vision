import cv2
import os
import argparse

from core.detector import DigitRecognizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--infile', type = str, default = 'input.jpg', help = 'Path to input image')
parser.add_argument('--outfile', type = str, default = 'output.jpg', help = 'Path to output image')
args = parser.parse_args()

model = DigitRecognizer()
img = cv2.imread(str(args.infile))
model.process_frame(img, str(args.outfile))
