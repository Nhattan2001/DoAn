import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from PIL import ImageTk, Image
from classification import Classification
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Image Classification")
        self.geometry('1600x1000')
        self.config(bg="skyblue")

        self.number_of_retrieval = 3

        self.create_panel_image()
        self.create_panel_image_classification()
        self.create_panel_feature_extraction()
        self.create_panel_image_retrieval()
        self.create_panel_options()

        self.classifier = Classification()

    def load_image_classification(self):
        self.image_path = filedialog.askopenfilename()
        img = Image.open(self.image_path)
        img = img.resize((270, 330), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img)
        if hasattr(self, 'frame_classification'):
            self.frame_classification.destroy()
        self.frame_classification = Label(self.panel_image, image=img)
        self.frame_classification.pack()
        self.frame_classification.image = img
        
        self.button_image_classification.config(state="normal")
        self.button_image_retrieval.config(state="normal")
        self.button_feature_extract.config(state="normal")
        
        return 0
        
    def create_panel_image(self):
        self.panel_image = Label(self, height=23, width=40, bg='grey', text="Image")
        self.panel_image.grid(rowspan=1, row=0)
    
    def create_panel_feature_extraction(self):
        self.panel_feature_extraction = Label(self, height=23, width=170, bg='gray', text="Feature Extraction")
        self.panel_feature_extraction.grid(rowspan=2, row=0, column=1, padx=10, pady=5)
        
    def create_panel_image_retrieval(self):
        self.panel_image_retrieval = Label(self, height=27, width=170, bg='gray', text="Image Retrieval")
        self.panel_image_retrieval.grid(row=2, column=1, padx=10, pady=5)
        
    def create_panel_options(self):
        self.left_frame = Frame(self, width=200, height=200, bg='gray')
        self.left_frame.grid(rowspan=2, row=2, padx=10, pady=5)
        
        self.panel_options = Label(self.left_frame, text="BẢNG CHỌN", anchor="center",
              bg='Cyan')
        self.panel_options.grid(row=0, column=0, padx=5, pady=5)
        
        self.tool_bar = Frame(self.left_frame, width=200, height=185)
        self.tool_bar.grid(row=2, column=0, padx=5, pady=5)
        
        self.button_image_load = Button(self.tool_bar, text='1.Image load', command=self.load_image_classification)
        self.button_image_load.grid(row=0, column=0, padx=5, pady=10, ipadx=10)

        self.button_feature_extract = Button(self.tool_bar, text="2.Feature extract", command=self.do_feature_extraction, state=DISABLED)
        self.button_feature_extract.grid(row=0, column=1, padx=5, pady=10, ipadx=10)
        
        self.button_image_classification = Button(self.tool_bar, text="3.Image classification", command = self.do_image_classification, state=DISABLED)
        self.button_image_classification.grid(row=1, column=0, padx=5, pady=10, ipadx=10)
        
        self.button_image_retrieval = Button(self.tool_bar, text="4.Image retrieval", command = self.do_image_retrevial, state=DISABLED)
        self.button_image_retrieval.grid(row=1, column=1, padx=5, pady=10, ipadx=10)
    
    def create_panel_image_classification(self):
        self.panel_image_classification = Label(self, height=2,width=20,bd=5, text="Prediction...")
        # self.panel_image_classification.grid(rowspan=3,row=0)
        self.panel_image_classification.place(x=70, y=400)

    def do_image_classification(self):
        if hasattr(self, 'image_path') == False:
            messagebox.showerror("Error", "Please load image first")
            return
        self.prediction_class = self.classifier.prediction(self.image_path)[0]
        self.panel_image_classification.config(text=self.prediction_class)
        
    def do_image_retrevial(self):
        retrieval_images = self.classifier.retriev_image_from_class(self.image_path)
        
        self.frame_image_retrievals = []
        for i in range(self.number_of_retrieval):
            try:
                img = Image.open(retrieval_images[i])
                img = img.resize((270, 330), Image.Resampling.LANCZOS)
                img = ImageTk.PhotoImage(img)
                self.frame_image_retrievals.append(Label(self.panel_image_retrieval, image=img))
                self.frame_image_retrievals[i].image = img        
                self.frame_image_retrievals[i].grid(row=0, column=i)
            except:
                pass
        
        return 0
    
    def do_feature_extraction(self):
        figs = self.classifier.visualize_feature_map(self.image_path)
        # plt figure to tkinter
        self.frame_feature_extraction = []
        for i in range(len(figs)):
            fig = figs[i]
            # resize figure
            fig.set_size_inches(3,3)
            canvas = FigureCanvasTkAgg(fig, self.panel_feature_extraction)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=i)
            self.frame_feature_extraction.append(canvas)
        return 0
        
if __name__ == "__main__":
    gui = GUI()
    
    
    
    gui.mainloop()
