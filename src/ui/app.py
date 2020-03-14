from tkinter import *
from ui.paint import Paint
from networks.neural_network import NeuralNetwork
from random import randint
import numpy


class App():

    def __init__(self, network=None, learning_rate=0.05):
        self.network = network
        self.network.set_learning_rate(learning_rate)

        self.init_window()

    def init_window(self):
        self.root = Tk()
        self.root.title("CursivePy")

        # Main Frame
        self.main_frame = Frame(self.root)
        self.main_frame.grid(row=0, column=0)

        # Label to indicate what number to draw
        self.target_text = Label(self.main_frame, text='')
        self.target_text.grid(row=0, column=0, columnspan=2, pady=(20, 0))
        self.target_text.config(font=("Arial", 24, 'bold'))
        self.set_target_number()

        # Left panel for painting
        self.paint_frame = Frame(self.main_frame)
        self.paint_frame.grid(
            row=1, column=0, padx=(30, 15), pady=30, sticky=W)
        self.paint = Paint(self.paint_frame, self.on_paint)

        self.clear_button = Button(self.paint_frame,
                                   width=35,
                                   cursor="pointinghand",
                                   text='Clear')
        self.clear_button.grid(row=1, column=0, padx=10, pady=10)
        self.clear_button.config(font=("Arial", 14, 'bold'))
        self.clear_button.bind("<ButtonPress>", self.on_clear_down)
        self.clear_button.bind("<ButtonRelease>", self.on_clear_up)

        self.submit_button = Button(self.paint_frame,
                                    width=35,
                                    cursor="pointinghand",
                                    text='Submit')
        self.submit_button.grid(row=2, column=0, padx=10, pady=10)
        self.submit_button.config(font=('Arial', 14, 'bold'))
        self.submit_button.bind("<ButtonPress>", self.on_submit_down)
        self.submit_button.bind("<ButtonRelease>", self.on_submit_up)

        # Left panel for ai prediction
        self.ai_frame = Frame(self.main_frame)
        self.ai_frame.grid(row=1, column=1, padx=(15, 30), pady=30, sticky=E)

        self.prediction_text = Label(self.ai_frame, text='AI Prediction')
        self.prediction_text.grid(row=0, column=0, pady=20)
        self.prediction_text.config(font=("Aril", 44))

        self.prediction_value = Label(self.ai_frame, text='')
        self.prediction_value.grid(row=1, column=0)
        self.prediction_value.config(font=("Aril", 44, 'bold'))

        self.center_window()

    def center_window(self):
        self.root.update_idletasks()
        windowWidth = self.root.winfo_width()
        windowHeight = self.root.winfo_height()

        positionRight = int(self.root.winfo_screenwidth() / 2
                            - windowWidth / 2)
        positionDown = int(self.root.winfo_screenheight() / 2
                           - windowHeight / 2)

        self.root.geometry("+{}+{}".format(positionRight, positionDown))

    def set_target_number(self):
        self.target = randint(0, 9)
        self.target_text.config(text="Draw a " + str(self.target))

    def on_paint(self, image_data):
        self.image_batch = image_data.reshape(1, pow(len(image_data), 2))
        prediction_batch = self.network.predict(self.image_batch)
        prediction_batch = self.network.convertOutputToResult(prediction_batch)
        self.set_prediction(prediction_batch[0])
        pass

    def set_prediction(self, prediction):
        if prediction == None:
            self.prediction_value.config(text="    ")
            return
        else:
            if prediction == self.target:
                self.prediction_value.config(fg="green")
            else:
                self.prediction_value.config(fg="red")
            self.prediction_value.config(text=prediction)

    def on_clear_down(self, event):
        self.clear_button.config(highlightbackground='#efefef')
        self.paint.clear()

    def on_clear_up(self, event):
        self.clear_button.config(highlightbackground='#ffffff')

    def submit(self):
        value_batch = numpy.array([self.target])
        solution_batch = self.network.convertResultToSolution(value_batch)
        self.network.train(self.image_batch, solution_batch)
        self.set_target_number()
        self.set_prediction(None)
        self.paint.clear()

    def on_submit_down(self, event):
        self.submit_button.config(highlightbackground='#efefef')
        self.submit()

    def on_submit_up(self, event):
        self.submit_button.config(highlightbackground='#ffffff')

    def start(self):
        self.root.mainloop()
