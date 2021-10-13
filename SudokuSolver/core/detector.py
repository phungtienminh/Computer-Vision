import tensorflow as tf
import numpy as np
import cv2
import imutils

from tensorflow import keras
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
from matplotlib import pyplot as plt

from utils.color import Color
from core.solver import Solver


class DigitRecognizer:
    def __init__(self):
        self.model = keras.models.load_model('printed_digit_model.h5')

    def predict_digit(self, img):
        return self.model.predict(img)

    def process_frame(self, frame, outfile):
        puzzle, warped = self.extract_board(frame)
        _, thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        thresh = clear_border(thresh)
        h, w = warped.shape
        h -= h % 9
        w -= w % 9

        step_y, step_x = h // 9, w // 9
        board = []

        for y in range(0, h, step_y):
            for x in range(0, w, step_x):
                roi = thresh[y : y + step_y - 1, x : x + step_x - 1]
                roi = cv2.resize(roi, (28, 28))
                roi = roi.astype(np.float32)
                roi /= 255.0
                roi = roi.reshape((1, 28, 28, 1))
                result = self.predict_digit(roi)
                result = np.argmax(result)
                board.append(result)

        board = np.asarray(board).reshape((9, 9))
        solved_board = Solver.solve(board.copy())

        if solved_board is None:
            cv2.imwrite(outfile, puzzle)
        else:
            for row in range(9):
                for col in range(9):
                    if board[row][col] != solved_board[row][col]:
                        x = col * step_x
                        y = row * step_y
                        cv2.putText(puzzle, f'{solved_board[row][col]}', (x + step_x // 4, y + step_y - step_y // 4), \
                                    cv2.FONT_HERSHEY_COMPLEX, 2, Color.BLUE, 3, cv2.LINE_AA)

            cv2.imwrite(outfile, puzzle)

    def extract_board(self, frame):
        # Convert to gray, apply blurring and threshold to get binary image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 3)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh = cv2.bitwise_not(thresh)

        # Remove noise
        kernel = np.ones((2, 2), np.uint8)
        for _ in range(3):
            thresh = cv2.erode(thresh, kernel)

        for _ in range(7):
            thresh = cv2.dilate(thresh, kernel)

        # Find contours
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)

        puzzle_contour = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                puzzle_contour = approx
                break

        puzzle = four_point_transform(frame, puzzle_contour.reshape((4, 2)))
        warped = four_point_transform(gray, puzzle_contour.reshape((4, 2)))
        return puzzle, warped

