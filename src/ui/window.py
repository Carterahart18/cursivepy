from tkinter import *
from ui.paint import Paint
from networks.neural_network import NeuralNetwork


class Window():

    def __init__(self, WIDTH=800, HEIGHT=600, network=None):
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.network = network

        self.init_window()

    def init_window(self):
        self.root = Tk()
        self.root.title("CursivePy")

        self.main_frame = Frame(self.root)
        self.main_frame.grid(row=0, column=0)

        self.paint_frame = Frame(self.main_frame)
        self.paint_frame.grid(
            row=0, column=0, padx=(30, 15), pady=30, sticky=W)
        paint = Paint(self.paint_frame, self.on_paint)

        self.ai_frame = Frame(self.main_frame)
        self.ai_frame.grid(row=0, column=1, padx=(15, 30), pady=30, sticky=E)

        self.prediction_text = Label(self.ai_frame, text='AI Prediction')
        self.prediction_text.grid(row=0, column=0, pady=20)
        self.prediction_text.config(font=("Ariel", 44))

        self.prediction_value = Label(self.ai_frame, text='')
        self.prediction_value.grid(row=1, column=0)
        self.prediction_value.config(font=("Ariel", 44))

        self.center_window()

    def center_window(self):
        windowWidth = self.root.winfo_reqwidth()
        windowHeight = self.root.winfo_reqheight()

        positionRight = int(
            self.root.winfo_screenwidth() / 2 - self.WIDTH / 2)
        positionDown = int(self.root.winfo_screenheight() /
                           2 - self.HEIGHT / 2)

        self.root.geometry("+{}+{}".format(positionRight, positionDown))

    def on_paint(self, image_data):
        image = image_data.reshape(1, pow(len(image_data), 2))
        prediction = self.network.predict(image)
        prediction = self.network.convertOutputToResult(prediction)
        self.prediction_value.config(text=prediction)
        pass

    def set_prediction(self, prediction):
        self.prediction_value.config(text=prediction)

    def start(self):
        self.root.mainloop()
