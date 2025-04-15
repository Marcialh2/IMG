import os
import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import pydicom
from skimage.filters import threshold_otsu, threshold_yen, gaussian
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import skimage.measure

# === Cargar datos DICOM ===
def load_dicom_series(dicom_folder):
    dicom_files = sorted([f for f in os.listdir(dicom_folder) if f.endswith(".dcm")])
    if not dicom_files:
        raise FileNotFoundError("No se encontraron archivos DICOM en la carpeta especificada.")
    slices = [pydicom.dcmread(os.path.join(dicom_folder, f)) for f in dicom_files]
    slices.sort(key=lambda x: x.InstanceNumber)
    pixel_data = np.stack([s.pixel_array for s in slices], axis=-1)
    return pixel_data, slices

# === Preprocesamiento ===
def preprocess_image(volume):
    volume = gaussian(volume, sigma=1)
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    return volume

# === Segmentación ===
def segment_image(volume, method="otsu"):
    if method == "otsu":
        threshold = threshold_otsu(volume)
    elif method == "yen":
        threshold = threshold_yen(volume)
    else:
        raise ValueError("Método de segmentación no válido. Use 'otsu' o 'yen'.")
    return volume > threshold

# === Visualización 3D ===
def visualize_3d(volume):
    verts, faces, _, _ = skimage.measure.marching_cubes(volume, level=0)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.7)
    mesh.set_facecolor((0.5, 0.5, 1))
    ax.add_collection3d(mesh)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(0, volume.shape[0])
    ax.set_ylim(0, volume.shape[1])
    ax.set_zlim(0, volume.shape[2])
    plt.tight_layout()
    plt.show()

# === Interfaz gráfica ===
class DICOMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Segmentación DICOM")

        self.volume = None
        self.seg_otsu = None
        self.seg_yen = None
        self.index = tk.IntVar(value=0)
        self.selected_3d_method = tk.StringVar(value="Yen")

        self.setup_ui()

    def setup_ui(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Button(frame, text="Cargar DICOM", command=self.load_dicom).pack(side=tk.TOP, pady=5)

        # Método de reconstrucción 3D
        method_frame = ttk.Frame(frame)
        method_frame.pack(side=tk.TOP, pady=5)
        ttk.Label(method_frame, text="Método para 3D:").pack(side=tk.LEFT)
        method_selector = ttk.Combobox(method_frame, textvariable=self.selected_3d_method, state="readonly")
        method_selector['values'] = ["Otsu", "Yen"]
        method_selector.pack(side=tk.LEFT, padx=5)

        ttk.Button(frame, text="Reconstrucción 3D", command=self.show_3d).pack(side=tk.TOP, pady=5)

        self.slider = ttk.Scale(frame, from_=0, to=0, orient=tk.HORIZONTAL, command=self.update_images)
        self.slider.pack(fill=tk.X, pady=5)

        self.fig, self.axes = plt.subplots(1, 3, figsize=(12, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_dicom(self):
        folder = filedialog.askdirectory()
        if not folder:
            return

        volume, slices = load_dicom_series(folder)
        volume = preprocess_image(volume)

        self.volume = volume
        self.seg_otsu = segment_image(volume, method="otsu")
        self.seg_yen = segment_image(volume, method="yen")

        self.slider.configure(to=volume.shape[2] - 1)
        self.index.set(volume.shape[2] // 2)
        self.slider.set(volume.shape[2] // 2)

        self.update_images()

    def update_images(self, event=None):
        if self.volume is None:
            return

        i = int(float(self.slider.get()))
        self.axes[0].imshow(self.volume[:, :, i], cmap='gray')
        self.axes[0].set_title("Original")
        self.axes[1].imshow(self.seg_otsu[:, :, i], cmap='gray')
        self.axes[1].set_title("Otsu")
        self.axes[2].imshow(self.seg_yen[:, :, i], cmap='gray')
        self.axes[2].set_title("Yen")

        for ax in self.axes:
            ax.axis('off')
        self.fig.tight_layout()
        self.canvas.draw()

    def show_3d(self):
        if self.selected_3d_method.get() == "Otsu" and self.seg_otsu is not None:
            visualize_3d(self.seg_otsu)
        elif self.selected_3d_method.get() == "Yen" and self.seg_yen is not None:
            visualize_3d(self.seg_yen)

if __name__ == '__main__':
    root = tk.Tk()
    app = DICOMApp(root)
    root.mainloop()
