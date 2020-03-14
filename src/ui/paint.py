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

        # Init Fields
        self.brush_size = 2
        self.image_data = zeros((IMAGE_SIZE, IMAGE_SIZE))
        self.prev_col = None
        self.prev_row = None
        self.WIDTH = self.root.winfo_width()
        self.HEIGHT = self.root.winfo_width()

        # Init Canvas
        self.canvas = Canvas(self.root,
                             bg='black',
                             width=CANVAS_SIZE,
                             height=CANVAS_SIZE)
        self.canvas.grid(row=0, column=0)
        self.canvas.bind('<B1-Motion>', self._on_paint_down)
        self.canvas.bind('<ButtonRelease-1>', self._on_paint_up)

    def draw_image(self):
        """
        Draws a pixelated image using canvas rectangles using the current
        IMAGE_SIZE x IMAGE_SIZE data.
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
        """
        Draws a brush dot with the diameter of the current brush_size at coordinates at
        (row, col)

        Parameters
        ----------
        row: the row in the IMAGE_SIZE x IMAGE_SIZE image to draw to
        col: the column in the IMAGE_SIZE x IMAGE_SIZE imge to draw to
        """
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
                    # Draw blurred edge. Never subtract from value
                    self.image_data[i][j] = \
                        max(radius + 1 - dist, self.image_data[i][j])

    def _on_paint_down(self, event):
        """
        Draws a brush stroke from the previous (row, col) coordinates to the current event
        coordinates.

        Parameters
        ----------
        event: the event object on the canvas containing the current (x, y) coordinates
        """
        num_fill_points = 10
        col = event.x / PIXEL_SIZE
        row = event.y / PIXEL_SIZE

        # If drawing out of bounds, quit
        if col >= IMAGE_SIZE or col < 0:
            return
        if row >= IMAGE_SIZE or row < 0:
            return

        # Draw a line from prev coords to current coords
        if self.prev_col != None and self.prev_row != None:
            col_diff = col - self.prev_col
            row_diff = row - self.prev_row

            dcol = col_diff / (num_fill_points + 1)
            drow = row_diff / (num_fill_points + 1)

            # Calculate points in line from prev coords to current coords
            pointSet = set([])
            for i in range(1, num_fill_points + 1):
                col_point = self.prev_col + dcol * i
                row_point = self.prev_row + drow * i
                pointSet.add((row_point, col_point))

            # Draw line
            for point in pointSet:
                self.brush(point[0], point[1])
        else:
            # Else simply draw a dot
            self.brush(row, col)

        self.prev_col = col
        self.prev_row = row

        self.draw_image()

        # Send image data to parent
        self.on_paint_callback(self.image_data)

    def clear(self):
        """
        Clears the images data and canvas of all data
        """
        self.canvas.delete("all")
        self.image_data = zeros((IMAGE_SIZE, IMAGE_SIZE))

    def _on_paint_up(self, event):
        """
        Resets the previous row and column now that painting has stopped
        """
        self.prev_col = None
        self.prev_row = None


if __name__ == '__main__':
    root = Tk()
    root.title("Handwritten Digit Recognition")

    def on_paint_callback(image):
        pass

    paint = Paint(root, on_paint_callback)

    root.mainloop()
