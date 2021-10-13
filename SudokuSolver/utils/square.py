from utils.color import Color


class Square:
    def __init__(self, x, y, size, number):
        self.x = x
        self.y = y
        self.size = size
        self.number = number
        self.number_color = Color.GREEN
        self.border_color = Color.GREEN
        self.border_thickness = 1
