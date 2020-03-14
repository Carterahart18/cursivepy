from tkinter import *
from numpy import zeros
from math import sqrt

CANVAS_SIZE = 300
IMAGE_SIZE = 28
PIXEL_SIZE = CANVAS_SIZE / IMAGE_SIZE


class Paint():

    def __init__(self, root, on_paint_callback):
        self.root = root
        self.on_paint_callback = on_paint_callback

        self.init_fields()
        self.init_canvas()

    def init_fields(self):
        self.brush_size = 2
        self.image_data = zeros((IMAGE_SIZE, IMAGE_SIZE))
        self.prev_col = None
        self.prev_row = None

    def init_canvas(self):
        self.WIDTH = self.root.winfo_width()
        self.HEIGHT = self.root.winfo_width()

        self.canvas = Canvas(self.root,
                             bg='black',
                             width=CANVAS_SIZE,
                             height=CANVAS_SIZE)
        self.canvas.grid(row=0, column=0)
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)

        self.clear_button = Button(self.root,
                                   width=30,
                                   cursor="pointinghand",
                                   text='Clear')
        self.clear_button.grid(row=1, column=0, padx=10, pady=10)
        self.clear_button.config(font='Arial 14 bold')
        self.clear_button.bind("<ButtonPress>", self.on_clear_down)
        self.clear_button.bind("<ButtonRelease>", self.on_clear_up)

    def draw_image(self):
        """
        Draws a pixelated image using canvas rectangles using the current
        IMAGE_SIZE x IMAGE_SIZE data
        """
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
        min_row = max(0, int(row - radius))
        max_row = min(28, int(row + radius + 1))
        min_col = max(0, int(col - radius))
        max_col = min(28, int(col + radius + 1))
        for i in range(min_row, max_row):
            for j in range(min_col, max_col):
                # Pythagorean theorum to get distance from center
                row_diff = i - row
                col_diff = j - col
                dist = sqrt(pow(row_diff, 2) + pow(col_diff, 2))

                if dist < radius:
                    self.image_data[i][j] = 1
                elif dist < radius + 1:
                    self.image_data[i][j] = \
                        max(radius + 1 - dist, self.image_data[i][j])

    def paint(self, event):
        num_fill_points = 10
        col = event.x / PIXEL_SIZE
        row = event.y / PIXEL_SIZE

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
                col_point = self.prev_col + dcol * i
                row_point = self.prev_row + drow * i
                pointSet.add((row_point, col_point))

            for point in pointSet:
                self.brush(point[0], point[1])
        else:
            self.brush(row, col)

        self.prev_col = col
        self.prev_row = row

        self.draw_image()

        # Send image data to parent
        self.on_paint_callback(self.image_data)

    def on_clear_down(self, event):
        self.canvas.delete("all")
        self.image_data = zeros((IMAGE_SIZE, IMAGE_SIZE))
        self.clear_button.config(highlightbackground='#efefef')

    def on_clear_up(self, event):
        self.clear_button.config(highlightbackground='#ffffff')

    def reset(self, event):
        self.prev_col = None
        self.prev_row = None


if __name__ == '__main__':
    root = Tk()
    root.title("Handwritten Digit Recognition")

    paint = Paint(root)

    root.mainloop()
