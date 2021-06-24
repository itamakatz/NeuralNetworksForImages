#! /usr/bin/env python3

# info:
#   https://www.python-course.eu/tkinter_events_binds.php
#   https://stackoverflow.com/questions/32289175/list-of-all-tkinter-events
#   https://stackoverflow.com/questions/45471847/tkinter-display-two-images-side-by-side
#   https://stackoverflow.com/questions/2603169/update-tkinter-label-from-variable
#   https://stackoverflow.com/questions/24849265/how-do-i-create-an-automatically-updating-gui-using-tkinter

import os
from os import listdir
from natsort import natsorted
from os.path import isfile, join, isdir
import tkinter as tk
from PIL import ImageTk,Image  
import time, threading

DEBUG = False
UPDATE_INTERVAL_SECS = 60

def view():
  global img1, img2, plot_image, canvas, i, onlyfiles, dir_path, root, Lb, Lb_current_index, plot_file, l_status, dirs_info_dic, l_current_dir, timer

  onlyfiles = None
  
  root = tk.Tk()
  root.title('Viewer for NN generators!')
  root.resizable(False, False)
  root.geometry("%dx%d" % (750, 600))
  canvas = tk.Canvas(root, width = 300, height = 300)  
  canvas.pack(padx=50, pady=10, fill=tk.BOTH, expand=True)

  l_current_dir = tk.StringVar()
  current_dir_label = tk.Label(root, textvariable=l_current_dir,  fg="black", font="Helvetica 10")
  current_dir_label.pack() 

  Lb = tk.Listbox(root)
  Lb.bind("<<ListboxSelect>>", show_dir_info)
  Lb.pack(padx=10,pady=10,fill=tk.BOTH,expand=True)

  l_status = tk.StringVar()
  status_label = tk.Label(root, textvariable=l_status,  fg="black", font="Helvetica 15")
  status_label.pack()

  root.bind("<Left>", back)
  root.bind("<Right>", next)
  root.bind("<Home>", home)
  root.bind("<Prior>", back100)
  root.bind("<Next>", next100)  
  root.bind("<End>", end)
  root.bind("r", update_dirs_info)
  root.bind("<Return>", enter)
  root.bind("<KP_Enter>", enter)
  root.bind("<Escape>", lambda x: root.destroy())
  root.bind("<F5>", reload_dirs)

  reload_dirs()

  timer = threading.Timer(UPDATE_INTERVAL_SECS, update_dirs_info)
  timer.start()
  root.mainloop() 
  timer.cancel()


def update_dirs_info(event=None):
  global img1, img2, plot_image, canvas, i, onlyfiles, dir_path, root, Lb, Lb_current_index, plot_file, l_status, dirs_info_dic, l_current_dir, timer
  try:
    for dir in dirs_info_dic:
      images_count = sum(1 for f in os.listdir(dir) if os.path.isfile(join(dir, f)) and 'iter' not in f and 'batch_count' in f)
      dirs_info_dic[dir] = (f"Images count = {images_count}. Size: {get_size(dir)//1000000} MB.")
      update_status(f"calculating dirs info | {i}/{len(dirs_info_dic)}")
    update_status("Finished updating dirs info")
  except:
    update_status("Could not updated dirs info")
  timer = threading.Timer(UPDATE_INTERVAL_SECS, update_dirs_info)
  timer.start()

def reload_dirs(event=None):
  global img1, img2, plot_image, canvas, i, onlyfiles, dir_path, root, Lb, Lb_current_index, plot_file, l_status, dirs_info_dic, l_current_dir, timer
  canvas.delete('all')
  l_current_dir.set(f"")
  if(onlyfiles is not None):
    onlyfiles.clear()
  onlydirs = [d for d in listdir(os.path.dirname(os.path.realpath(__file__))) if isdir(join(os.path.dirname(os.path.realpath(__file__)), d))]
  onlydirs = natsorted(onlydirs)
  dirs_info_dic = {}
  Lb.delete(0,'end')
  for i in range(len(onlydirs)):
    Lb.insert(i+1, onlydirs[i])
    images_count = sum(1 for f in os.listdir(onlydirs[i]) if os.path.isfile(join(onlydirs[i], f)) and 'iter' not in f and 'batch_count' in f)
    dirs_info_dic[onlydirs[i]] = (f"Images count = {images_count}. Size: {get_size(onlydirs[i])//1000000} MB.")
    update_status(f"calculating dirs info | {i}/{len(onlydirs)}")
  update_status("Done!")
  img1 = None
  img2 = None
  plot_image = None
  Lb.focus_set()
  update_status("Please select a dir")
def show():
  global img1, img2, plot_image, canvas, i, onlyfiles, dir_path, root, Lb, Lb_current_index, plot_file, l_status, dirs_info_dic, l_current_dir, timer
  pilIm = Image.open(onlyfiles[i])
  update_status(f"Showing {i+1}/{len(onlyfiles)}")
  w, h = pilIm.size
  img1 = ImageTk.PhotoImage(pilIm)
  canvas.create_image((w/2, h/2),image=img1)
  if(plot_image is None):
    plot_image = Image.open(plot_file)
    plot_image = plot_image.resize((w, h), Image.ANTIALIAS)
    img2 = ImageTk.PhotoImage(plot_image)
    create_image2 = canvas.create_image((w/2, h/2), image=img2)
    canvas.move(create_image2, 340, 0)
def change_index(change):
  global img1, img2, plot_image, canvas, i, onlyfiles, dir_path, root, Lb, Lb_current_index, plot_file, l_status, dirs_info_dic, l_current_dir, timer
  if(i+change < len(onlyfiles) and i+change >= 0): i=i+change
  elif(i+change >= len(onlyfiles)): i = len(onlyfiles)-1
  elif(i+change < 0): i = 0
def next(event=None):
  global img1, img2, plot_image, canvas, i, onlyfiles, dir_path, root, Lb, Lb_current_index, plot_file, l_status, dirs_info_dic, l_current_dir, timer
  change_index(+1)
  show()
def back(event=None):
  global img1, img2, plot_image, canvas, i, onlyfiles, dir_path, root, Lb, Lb_current_index, plot_file, l_status, dirs_info_dic, l_current_dir, timer
  change_index(-1)
  show()
def next100(event=None):
  global img1, img2, plot_image, canvas, i, onlyfiles, dir_path, root, Lb, Lb_current_index, plot_file, l_status, dirs_info_dic, l_current_dir, timer
  change_index(+100)
  keep_listbox_index()
  show()
def back100(event=None):
  global img1, img2, plot_image, canvas, i, onlyfiles, dir_path, root, Lb, Lb_current_index, plot_file, l_status, dirs_info_dic, l_current_dir, timer
  change_index(-100)
  keep_listbox_index()
  show()  
def home(event=None):
  global img1, img2, plot_image, canvas, i, onlyfiles, dir_path, root, Lb, Lb_current_index, plot_file, l_status, dirs_info_dic, l_current_dir, timer
  i = 0
  keep_listbox_index()
  show()
def end(event=None):
  global img1, img2, plot_image, canvas, i, onlyfiles, dir_path, root, Lb, Lb_current_index, plot_file, l_status, dirs_info_dic, l_current_dir, timer
  i = len(onlyfiles)-1
  keep_listbox_index()
  show()
def reload(event=None):
  global img1, img2, plot_image, canvas, i, onlyfiles, dir_path, root, Lb, Lb_current_index, plot_file, l_status, dirs_info_dic, l_current_dir, timer
  update_status("loading images")
  onlyfiles = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f)) and join(dir_path, f).endswith(".png") and "iter" not in f and "batch_count" in f]
  plots_files = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f)) and join(dir_path, f).endswith(".png") and "iter" in f and "plot" in f]
  if(len(onlyfiles) == 0): 
    update_status("No images found..")
    return
  if(len(plots_files) == 0): 
    update_status("No plots found..")
    return    
  update_status("sorting plots")
  plots_files = natsorted(plots_files)
  plot_file = plots_files[-1]
  update_status("sorting images")
  onlyfiles = natsorted(onlyfiles)
  update_status("sort finished")
  i = len(onlyfiles)-1
def show_dir_info(event=None):
  global img1, img2, plot_image, canvas, i, onlyfiles, dir_path, root, Lb, Lb_current_index, plot_file, l_status, dirs_info_dic, l_current_dir, timer
  Lb_current_index = Lb.curselection()
  dir = Lb.get(Lb_current_index)
  if (dir not in dirs_info_dic):
    images_count = sum(1 for f in os.listdir(dir) if os.path.isfile(join(dir, f)) and 'iter' not in f and 'batch_count' in f)
    dirs_info_dic[dir] = (f"Images count = {images_count}. Size: {get_size(dir)//1000} kB.")
  update_status(dirs_info_dic[dir])
def enter(event=None):
  global img1, img2, plot_image, canvas, i, onlyfiles, dir_path, root, Lb, Lb_current_index, plot_file, l_status, dirs_info_dic, l_current_dir, timer
  dir = Lb.get(Lb.curselection())
  l_current_dir.set(f"Current Dir: {dir}")
  root.update_idletasks()
  dir_path = join(os.path.dirname(os.path.realpath(__file__)), dir)
  plot_image = None
  reload()
  end()
def update_status(msg):
  global img1, img2, plot_image, canvas, i, onlyfiles, dir_path, root, Lb, Lb_current_index, plot_file, l_status, dirs_info_dic, l_current_dir, timer
  l_status.set(msg)
  if(DEBUG): print(msg)
  root.update_idletasks()
def keep_listbox_index():
  Lb.select_set(Lb_current_index)
  Lb.activate(Lb_current_index)

def get_size(start_path = '.'):
  total_size = 0
  for dirpath, dirnames, filenames in os.walk(start_path):
    for f in filenames:
      fp = os.path.join(dirpath, f)
      # skip if it is symbolic link
      if not os.path.islink(fp):
        total_size += os.path.getsize(fp)

  return total_size

if __name__ == '__main__':
  os.chdir(os.path.dirname(os.path.realpath(__file__)))
  print("Starting")
  view()
  print("Finished")