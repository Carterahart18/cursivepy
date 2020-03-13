from tkinter import *
from tkinter.colorchooser import askcolor
import numpy
import random
import math
from PIL import ImageGrab, Image, ImageDraw

CANVAS_SIZE = 280
IMAGE_SIZE = 28
PIXEL_SIZE = CANVAS_SIZE / IMAGE_SIZE


class Paint(object):

    def __init__(self):
        self.image_data = numpy.zeros((IMAGE_SIZE, IMAGE_SIZE))

        self.setup_frame()
        self.setup_fields()
        self.root.mainloop()

    def setup_frame(self):
        self.root = Tk()

        self.clear_button = Button(self.root, text='Clear', command=self.clear)
        self.clear_button.grid(row=0, column=0)

        self.canvas = Canvas(self.root,
                             bg='black',
                             width=CANVAS_SIZE,
                             height=CANVAS_SIZE)

        self.canvas.grid(row=1, columnspan=5)

    def setup_fields(self):
        self.prev_col = None
        self.prev_row = None
        self.brush_size = 2
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)

    def draw_image(self, event):
        self.canvas.delete("all")
        for row in range(IMAGE_SIZE):
            for col in range(IMAGE_SIZE):
                value = int(self.image_data[row][col] * 255)
                value = '%02x' % value
                value = '#' + str(value) + str(value) + str(value)

                self.canvas.create_rectangle(col * PIXEL_SIZE,
                                             row * PIXEL_SIZE,
                                             col * PIXEL_SIZE + PIXEL_SIZE,
                                             row * PIXEL_SIZE + PIXEL_SIZE,
                                             fill=value,
                                             outline=value)

    def brush(self, row, col):
        radius = self.brush_size / 2
        min_row = int(row - radius)
        max_row = int(row + radius + 1)
        min_col = int(col - radius)
        max_col = int(col + radius + 1)
        for i in range(min_row, max_row):
            for j in range(min_col, max_col):
                # Pythagorean theorum to get distance from center
                row_diff = i - row
                col_diff = j - col
                dist = math.sqrt(pow(row_diff, 2) + pow(col_diff, 2))

                if dist < radius:
                    self.image_data[i][j] = 1
                elif dist < radius + 1:
                    self.image_data[i][j] = \
                        max(radius + 1 - dist, self.image_data[i][j])
        pass

    def paint(self, event):
        num_fill_points = 10
        col = int(event.x // PIXEL_SIZE)
        row = int(event.y // PIXEL_SIZE)

        if col >= IMAGE_SIZE or col < 0:
            return
        if row >= IMAGE_SIZE or row < 0:
            return

        if self.prev_col != None and self.prev_row != None:
            col_diff = col - self.prev_col
            row_diff = row - self.prev_row

            dcol = col_diff / (num_fill_points + 1)
            drow = row_diff / (num_fill_points + 1)

            pointSet = set([])
            for i in range(1, num_fill_points + 1):
                col_point = int(self.prev_col + dcol * i)
                row_point = int(self.prev_row + drow * i)
                pointSet.add((row_point, col_point))

            for point in pointSet:
                self.brush(point[0], point[1])
                self.image_data[point[0]][point[1]] = 1
        else:
            self.brush(row, col)

        self.prev_col = col
        self.prev_row = row

        self.draw_image(event)

    def clear(self):
        self.canvas.delete("all")
        self.image_data = numpy.zeros((IMAGE_SIZE, IMAGE_SIZE))

    def reset(self, event):
        self.prev_col = None
        self.prev_row = None


if __name__ == '__main__':
    Paint()
