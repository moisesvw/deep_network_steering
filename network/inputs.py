import tkinter as tk
from PIL import ImageTk, Image

root1 = tk.Tk()
root1.title("Current frame center camera")
root1.geometry("320x160")
img1 = ImageTk.PhotoImage(Image.open("current_frame.jpg"))
c1 = tk.Canvas(root1)
c1.pack(side='top', fill='both', expand='yes')
picture_one = c1.create_image(0, 0, image=img1, anchor='nw')


root2 = tk.Toplevel()
root2.title("Cropping")
root2.geometry("320x70")
ii = Image.open("current_crop.jpg").convert('RGB')
img2 = ImageTk.PhotoImage(ii)
c2 = tk.Canvas(root2)
c2.pack(side='top', fill='both', expand='yes')
picture_two = c2.create_image(0, 0, image=img2, anchor='nw')
while True:
    try:
        picture3 = ImageTk.PhotoImage(Image.open("current_frame.jpg"))
        c1.itemconfigure(picture_one, image = picture3)
        root1.update()

        ii = Image.open("current_crop.jpg").convert('RGB')
        picture3_ = ImageTk.PhotoImage(ii)
        c2.itemconfigure(picture_two, image = picture3_)
        root2.update()
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(e)
