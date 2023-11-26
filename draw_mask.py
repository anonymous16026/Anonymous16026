import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageTk

class ImageMaskEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Mask Editor")

        self.image_path = ""
        self.image = None
        self.mask = None
        self.mask_history = []
        self.image_history = []

        # Canvas for displaying image and drawing mask
        self.canvas = tk.Canvas(root)
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)
        self.canvas.bind("<B1-Motion>", self.draw_mask)
        #self.canvas.bind("<a>", self.undo)  # Bind Ctrl+Z for undo

        # Menu bar
        menubar = tk.Menu(root)
        root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_command(label="Save Mask", command=self.save_mask)
        root.bind_all("<Control-z>", self.undo)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])
        if file_path:
            self.image_path = file_path
            self.image = Image.open(file_path)
            self.mask = Image.new("L", self.image.size, color=0)
            self.mask_history = [self.mask.copy()]
            self.image_history = [self.image.copy()]
            self.display_image()

    def display_image(self):
        if self.image:
            tk_image = ImageTk.PhotoImage(self.image)
            self.canvas.config(width=tk_image.width(), height=tk_image.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
            self.canvas.image = tk_image

    def draw_mask(self, event):
        if self.image:
            x, y = event.x, event.y
            draw = ImageDraw.Draw(self.mask)
            draw_image = ImageDraw.Draw(self.image)
            draw.ellipse([(x - 5, y - 5), (x + 5, y + 5)], fill=255, outline=255)
            draw_image.ellipse([(x - 5, y - 5), (x + 5, y + 5)], fill=255, outline=255)
            if len(self.mask_history) > 500:
                self.mask_history.pop(0)
                self.image_history.pop(0)
            self.mask_history.append(self.mask.copy())
            self.image_history.append(self.image.copy())
            self.display_image()

    def undo(self, event):
        if len(self.mask_history) > 1:
            self.mask_history.pop()
            self.image_history.pop()
            self.mask = self.mask_history[-1].copy()
            self.image = self.image_history[-1].copy()
            self.display_image()

    def save_mask(self):
        if self.mask:
            mask_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if mask_path:
                self.mask.save(mask_path)
                print(f"Mask saved to {mask_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageMaskEditor(root)
    root.mainloop()
