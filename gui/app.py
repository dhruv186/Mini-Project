import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
from sorter.face_sorter import FaceSorter


def main():
    root = tk.Tk()
    root.title('Face Sorter')
    root.geometry('760x560')

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill='both', expand=True)

    ttk.Label(frm, text='Source folder (new photos):').grid(column=0, row=0, sticky='w')
    src_var = tk.StringVar()
    ttk.Entry(frm, textvariable=src_var, width=60).grid(column=1, row=0, sticky='we', padx=5)
    ttk.Button(frm, text='Browse', command=lambda: src_var.set(filedialog.askdirectory())).grid(column=2, row=0)

    ttk.Label(frm, text='Sorted folder (output):').grid(column=0, row=1, sticky='w')
    out_var = tk.StringVar()
    ttk.Entry(frm, textvariable=out_var, width=60).grid(column=1, row=1, sticky='we', padx=5)
    ttk.Button(frm, text='Browse', command=lambda: out_var.set(filedialog.askdirectory())).grid(column=2, row=1)

    ttk.Label(frm, text='DBSCAN eps:').grid(column=0, row=2, sticky='w')
    eps_var = tk.DoubleVar(value=0.55)
    ttk.Entry(frm, textvariable=eps_var, width=10).grid(column=1, row=2, sticky='w')
    ttk.Label(frm, text='min_samples:').grid(column=1, row=2, sticky='e')
    min_samples_var = tk.IntVar(value=1)
    ttk.Entry(frm, textvariable=min_samples_var, width=6).grid(column=1, row=2, sticky='e', padx=(100,0))

    ttk.Label(frm, text='Face model:').grid(column=0, row=3, sticky='w')
    model_var = tk.StringVar(value='hog')
    ttk.OptionMenu(frm, model_var, 'hog', 'hog', 'cnn').grid(column=1, row=3, sticky='w')
    move_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(frm, text='Move files (instead of copy)', variable=move_var).grid(column=2, row=3)

    logbox = scrolledtext.ScrolledText(frm, width=80, height=16)
    logbox.grid(column=0, columnspan=3, row=6, pady=6)

    progress = ttk.Progressbar(frm, orient='horizontal', mode='determinate')
    progress.grid(column=1, row=4, sticky='we', padx=5)

    def log(msg):
        logbox.insert('end', msg + '\n')
        logbox.see('end')

    def start():
        src = src_var.get().strip()
        out = out_var.get().strip()
        if not src or not out:
            messagebox.showerror('Error', 'Please choose source and output folders.')
            return
        sorter = FaceSorter(src, out, eps=eps_var.get(), min_samples=min_samples_var.get(), model=model_var.get(), move_files=move_var.get(), logger=log)
        t = threading.Thread(target=lambda: sorter.run(), daemon=True)
        t.start()

    ttk.Button(frm, text='Start Sorting', command=start).grid(column=0, row=4)

    frm.columnconfigure(1, weight=1)
    root.mainloop()