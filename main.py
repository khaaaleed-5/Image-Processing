import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from utils import read_image, threshold, simple_halftoning, advanced_halftoning, histogram, simple_edge_sobel, simple_edge_prewitt, simple_edge_kirsch, advanced_edge_homogeneity, advanced_edge_difference, advanced_edge_differenceofGaussians, advanced_edge_contrastBased, low_bass_filtering, high_bass_filtering, add_image, subtract_image, invert_image, histogram_segementation

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pixel Studio")
        self.root.geometry("1300x700")
        self.root.configure(bg="#EAEDED") 

        # Button and font styles
        self.button_bg = "#007ACC"
        self.button_fg = "#FFFFFF"
        self.font = ("Arial", 12, "bold")

        # Frames
        self.control_frame = tk.Frame(root, bg="#EAEDED")
        self.control_frame.pack(side=tk.LEFT, padx=20, pady=20, anchor="nw")

        self.image_frame = tk.Frame(root, bg="#FFFFFF", bd=2, relief=tk.SOLID)
        self.image_frame.pack(side=tk.RIGHT, padx=20, pady=20, expand=True, fill=tk.BOTH)

        # Upload button
        self.upload_button = tk.Button(
            self.control_frame,
            text="Upload Photo",
            command=self.upload_photo,
            bg=self.button_bg,
            fg=self.button_fg,
            font=self.font,
            relief="flat",  # Make the button's edges flat
            width=20,
            pady=10,  
            height=2 
        )
        self.upload_button.pack(pady=10)

        # Functionality frame
        self.functionality_frame = tk.Frame(self.control_frame, bg="#EAEDED")
        self.functionality_frame.pack(pady=20)

        # Image attributes
        self.original_image = None
        self.processed_image = None

        # Halftoning dropdown menu 
        self.halftoning_var = tk.StringVar(self.control_frame)
        self.halftoning_var.set("Halftoning")
        
        # Simple edge dropdown menu
        self.simple_edge_detection_var = tk.StringVar(self.control_frame)
        self.simple_edge_detection_var.set("Simple Edge")

        # Advanced edge dropdown menu
        self.advanced_edge_detection_var = tk.StringVar(self.control_frame)
        self.advanced_edge_detection_var.set("Advanced Edge")

        # Filtering dropdown menu
        self.filtering_var = tk.StringVar(self.control_frame)
        self.filtering_var.set("Filtering")

        # Operations dropdown menu
        self.operation_var = tk.StringVar(self.control_frame)
        self.operation_var.set("Operations")

        # Histogram segementation dropdown menu
        self.histseg_var = tk.StringVar(self.control_frame)
        self.histseg_var.set("Segementation")


    def upload_photo(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if file_path:
            # Load the image by read function from utils
            self.original_image = read_image(file_path)
            self.display_images()

            # Add functionality buttons
            self.add_functionality_buttons()

    def add_functionality_buttons(self):
        for widget in self.functionality_frame.winfo_children():
            widget.destroy()

        self.create_function_button("Threshold", self.threshold)
        self.create_function_button("Histogram", self.histogram)

        
        # Halftoning dropdown menu 
        halftoning_menu = tk.OptionMenu(
            self.functionality_frame,
            self.halftoning_var,
            "Simple Halftoning", "Advanced Halftoning"
        )
        halftoning_menu.config(
            bg=self.button_bg, fg=self.button_fg, font=self.font, width=20,
            height=1  # Set fixed height to match buttons
        )
        halftoning_menu["menu"].config(
            bg=self.button_bg, fg=self.button_fg
        )  # Style the dropdown menu 
        halftoning_menu.pack(pady=10)

        # Trigger halftoning function based on selected option
        self.halftoning_var.trace("w", self.on_halftoning_select)

        # Simple Edge detection dropdown menu 
        simple_edge_detection_menu = tk.OptionMenu(
            self.functionality_frame,
            self.simple_edge_detection_var,
            "Sobel", "Prewitt", "Kirsch"
        )
        simple_edge_detection_menu.config(
            bg=self.button_bg, fg=self.button_fg, font=self.font, width=20,
            height=1 
        )
        simple_edge_detection_menu["menu"].config(
            bg=self.button_bg, fg=self.button_fg
        )  # Style the dropdown menu
        simple_edge_detection_menu.pack(pady=10)

        # Trigger edge detection function based on selected option
        self.simple_edge_detection_var.trace("w", self.on_simple_edge_detection_select)


        # Advanced Edge detection dropdown menu
        advanced_edge_detection_menu = tk.OptionMenu(
            self.functionality_frame,
            self.advanced_edge_detection_var,
            "Homogeneity", "Difference", "Gaussian", "Contrast-based"
        )
        advanced_edge_detection_menu.config(
            bg=self.button_bg, fg=self.button_fg, font=self.font, width=20,
            height=1  # Set fixed height to match buttons
        )
        advanced_edge_detection_menu["menu"].config(
            bg=self.button_bg, fg=self.button_fg
        )  # Style the dropdown menu 
        advanced_edge_detection_menu.pack(pady=10)

        # Trigger edge detection function based on selected option
        self.advanced_edge_detection_var.trace("w", self.on_advanced_edge_detection_select)

        # filtering dropdown menu
        filtering_menu = tk.OptionMenu(
            self.functionality_frame,
            self.filtering_var,
            "High-Bass", "Low-Bass"
        )
        filtering_menu.config(
            bg=self.button_bg, fg=self.button_fg, font=self.font, width=20,
            height=1  # Set fixed height to match buttons
        )
        filtering_menu["menu"].config(
            bg=self.button_bg, fg=self.button_fg
        )  # Style the dropdown menu 
        filtering_menu.pack(pady=10)

        # Trigger filtering function based on selected option
        self.filtering_var.trace("w", self.on_filtering_select)

        # Operations dropdown menu
        Op_menu = tk.OptionMenu(
            self.functionality_frame,
            self.operation_var,
            "Add", "Subtract","Invert"
        )
        Op_menu.config(
            bg=self.button_bg, fg=self.button_fg, font=self.font, width=20,
            height=1
        )
        Op_menu["menu"].config(
            bg=self.button_bg, fg=self.button_fg
        )  # Style the dropdown menu 
        Op_menu.pack(pady=10)

        # Trigger filtering function based on selected option
        self.operation_var.trace("w", self.on_operation_select)

     # histogram segementation dropdown menu
        histseg_menu = tk.OptionMenu(
            self.functionality_frame,
            self.histseg_var,
            "Peak", "Valley", "Adaptive"
        )
        histseg_menu.config(
            bg=self.button_bg, fg=self.button_fg, font=self.font, width=20,
            height=1
        )
        histseg_menu["menu"].config(
            bg=self.button_bg, fg=self.button_fg
        )  # Style the dropdown menu 
        histseg_menu.pack(pady=10)

        # Trigger filtering function based on selected option
        self.operation_var.trace("w", self.on_histseg_select)





    def create_function_button(self, text, command):
        button = tk.Button(
            self.functionality_frame,
            text=text,
            command=command,
            bg=self.button_bg,
            fg=self.button_fg,
            font=self.font,
            relief="flat",  # Flat button style
            width=20,
            pady=10,  
            height=1  
        )
        button.pack(pady=10)

    def on_halftoning_select(self, *args):
        halftoning_type = self.halftoning_var.get()
        if halftoning_type == "Simple Halftoning":
            self.simple_halftoning()
        elif halftoning_type == "Advanced Halftoning":
            self.advanced_halftoning()
    
    def on_simple_edge_detection_select(self, *args):
        edge_detection_type = self.simple_edge_detection_var.get()
        if edge_detection_type == "Sobel":
            self.simple_edge_sobel()
        elif edge_detection_type == "Prewitt":
            self.simple_edge_prewitt()
        elif edge_detection_type == "Kirsch":
            self.simple_edge_kirsch()
    
    def on_advanced_edge_detection_select(self, *args):
        edge_detection_type = self.advanced_edge_detection_var.get()
        if edge_detection_type == "Homogeneity":
            self.advanced_edge_homogeneity()
        elif edge_detection_type == "Difference":
            self.advanced_edge_difference()
        elif edge_detection_type == "Gaussian":
            self.advanced_edge_differenceofGaussians()
        else:
            self.advanced_edge_contrastBased()
        
    def on_filtering_select(self, *args):
        filtering_type = self.filtering_var.get()
        if filtering_type == "High-Bass":
            self.high_bass_filtering()
        elif filtering_type == "Low-Bass":
            self.low_bass_filtering()
    
    def on_operation_select(self, *args):
        op_type = self.operation_var.get()
        if op_type == "Add":
            self.add_image()
        elif op_type == "Subtract":
            self.subtract_image()
        elif op_type == "Invert":
            self.Invert_image()
    
    def on_histseg_select(self, *args):
        op_type = self.histseg_var.get()
        #####


    def display_images(self):
        # Clear the image frame
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        # Display the original image
        if self.original_image is not None:
            original_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            original_pil = Image.fromarray(original_img)
            original_resized = original_pil.resize((400, 400))
            original_photo = ImageTk.PhotoImage(original_resized)
            original_label = tk.Label(self.image_frame, text="Original Image", bg="#FFFFFF", font=self.font)
            original_label.grid(row=0, column=0, padx=10, pady=5)
            original_image_label = tk.Label(self.image_frame, image=original_photo, bg="#FFFFFF")
            original_image_label.image = original_photo
            original_image_label.grid(row=1, column=0, padx=10, pady=5)

        # Display the processed image
        if self.processed_image is not None:
            processed_pil = Image.fromarray(self.processed_image)
            processed_resized = processed_pil.resize((400, 400))
            processed_photo = ImageTk.PhotoImage(processed_resized)
            processed_label = tk.Label(self.image_frame, text="Processed Image", bg="#FFFFFF", font=self.font)
            processed_label.grid(row=0, column=1, padx=10, pady=5)
            processed_image_label = tk.Label(self.image_frame, image=processed_photo, bg="#FFFFFF")
            processed_image_label.image = processed_photo
            processed_image_label.grid(row=1, column=1, padx=10, pady=5)

    def threshold(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the threshold function from image_utils
        threshold_value = threshold(self.original_image)

        # Display threshold value
        messagebox.showinfo("Threshold Applied", f"Threshold value: {threshold_value:.2f}")

    def simple_halftoning(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the halftoning function from image_utils
        processed_image = simple_halftoning(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images()

    def advanced_halftoning(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the advanced halftoning function from image_utils
        processed_image = advanced_halftoning(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images()
    
    def histogram(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the histogram function from utils
        processed_image = histogram(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images()
    
    def simple_edge_sobel(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the simeple edge sobel function from utils
        processed_image = simple_edge_sobel(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images()    

    def simple_edge_prewitt(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the simple edge prewitt function from utils
        processed_image = simple_edge_prewitt(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images()

    def simple_edge_kirsch(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the advanced halftoning function from image_utils
        processed_image = simple_edge_kirsch(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images()

    def advanced_edge_homogeneity(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the advanced edge homogeneity function from image_utils
        processed_image = advanced_edge_homogeneity(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images()

    def advanced_edge_difference(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the advanced edge difference function from utils
        processed_image = advanced_edge_difference(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images()
    
    def advanced_edge_differenceofGaussians(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the advanced edge difference function from utils
        processed_image = advanced_edge_differenceofGaussians(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images()
    
    def advanced_edge_contrastBased(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the advanced edge contrast-based function from utils
        processed_image = advanced_edge_contrastBased(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images()
    
    def high_bass_filtering(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the high bass function from utils
        processed_image = high_bass_filtering(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images()

    def low_bass_filtering(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the high bass function from utils
        processed_image = low_bass_filtering(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images()
    
    def add_image(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the add image function from utils
        processed_image = add_image(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images()
    
    def subtract_image(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the subtract image function from utils
        processed_image = subtract_image(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images()


    def Invert_image(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the subtract image function from utils
        processed_image = invert_image(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images()

    def histogram_segmentation(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the advanced histogram function from utils
        processed_image = histogram_segementation(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images()

if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()
