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
        self.pen_size = 5
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)

    def clear(self):
        self.canvas.delete("all")
        self.image_data = numpy.zeros((IMAGE_SIZE, IMAGE_SIZE))

    def paint(self, event):
        x = int(event.x // PIXEL_SIZE)
        y = int(event.y // PIXEL_SIZE)

        self.image_data[y][x] = 1

        self.draw_image(event)

        self.prev_x = x
        self.prev_y = y

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

    def reset(self, event):
        self.prev_x = None
        self.prev_y = None


if __name__ == '__main__':
    Paint()
