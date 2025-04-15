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
import multiprocessing as mp
from functools import partial
import gc

# === Cargar datos DICOM ===
def load_dicom_series(dicom_folder):
    dicom_files = sorted([f for f in os.listdir(dicom_folder) if f.endswith(".dcm")])
    if not dicom_files:
        raise FileNotFoundError("No se encontraron archivos DICOM en la carpeta especificada.")
    
    # Usar un pool de procesos para cargar los archivos DICOM
    with mp.Pool(processes=mp.cpu_count()) as pool:
        slices = pool.map(lambda f: pydicom.dcmread(os.path.join(dicom_folder, f)), dicom_files)
    
    slices.sort(key=lambda x: x.InstanceNumber)
    # Usar memmap para manejar grandes volúmenes de datos
    pixel_data = np.stack([s.pixel_array for s in slices], axis=-1)
    return pixel_data, slices

# === Preprocesamiento ===
def preprocess_slice(slice_data, sigma=1):
    slice_data = gaussian(slice_data, sigma=sigma)
    slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))
    return slice_data

def preprocess_image(volume):
    # Procesar cada slice en paralelo
    with mp.Pool(processes=mp.cpu_count()) as pool:
        processed_slices = pool.map(preprocess_slice, [volume[:,:,i] for i in range(volume.shape[2])])
    
    # Reconstruir el volumen
    processed_volume = np.stack(processed_slices, axis=2)
    return processed_volume

# === Segmentación ===
def segment_slice(slice_data, method="otsu", manual_threshold=None):
    if manual_threshold is not None:
        return slice_data > manual_threshold
    elif method == "otsu":
        threshold = threshold_otsu(slice_data)
    elif method == "yen":
        threshold = threshold_yen(slice_data)
    else:
        raise ValueError("Método de segmentación no válido. Use 'otsu' o 'yen'.")
    return slice_data > threshold

def segment_image(volume, method="otsu", manual_threshold=None):
    # Procesar cada slice en paralelo
    with mp.Pool(processes=mp.cpu_count()) as pool:
        segment_func = partial(segment_slice, method=method, manual_threshold=manual_threshold)
        segmented_slices = pool.map(segment_func, [volume[:,:,i] for i in range(volume.shape[2])])
    
    # Reconstruir el volumen segmentado
    segmented_volume = np.stack(segmented_slices, axis=2)
    return segmented_volume

# === Visualización 3D ===
def visualize_3d(volume):
    # Liberar memoria antes de la visualización 3D
    gc.collect()
    
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
        
        # Variables para los umbrales
        self.otsu_threshold = tk.DoubleVar(value=0.5)
        self.yen_threshold = tk.DoubleVar(value=0.5)
        
        # Configurar el número de procesos
        self.num_processes = mp.cpu_count()
        
        self.setup_ui()

    def setup_ui(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # Añadir información sobre el procesamiento
        info_frame = ttk.Frame(frame)
        info_frame.pack(side=tk.TOP, pady=5)
        ttk.Label(info_frame, text=f"Procesamiento usando {self.num_processes} núcleos").pack(side=tk.LEFT)

        ttk.Button(frame, text="Cargar DICOM", command=self.load_dicom).pack(side=tk.TOP, pady=5)

        # Método de reconstrucción 3D
        method_frame = ttk.Frame(frame)
        method_frame.pack(side=tk.TOP, pady=5)
        ttk.Label(method_frame, text="Método para 3D:").pack(side=tk.LEFT)
        method_selector = ttk.Combobox(method_frame, textvariable=self.selected_3d_method, state="readonly")
        method_selector['values'] = ["Otsu", "Yen"]
        method_selector.pack(side=tk.LEFT, padx=5)

        # Sliders para umbrales
        threshold_frame = ttk.Frame(frame)
        threshold_frame.pack(side=tk.TOP, pady=5)
        
        # Slider para Otsu
        otsu_frame = ttk.Frame(threshold_frame)
        otsu_frame.pack(side=tk.LEFT, padx=10)
        ttk.Label(otsu_frame, text="Umbral Otsu:").pack(side=tk.TOP)
        self.otsu_slider = ttk.Scale(otsu_frame, from_=0, to=1, orient=tk.HORIZONTAL, 
                                   variable=self.otsu_threshold, command=self.update_segmentation)
        self.otsu_slider.pack(side=tk.TOP)
        
        # Slider para Yen
        yen_frame = ttk.Frame(threshold_frame)
        yen_frame.pack(side=tk.LEFT, padx=10)
        ttk.Label(yen_frame, text="Umbral Yen:").pack(side=tk.TOP)
        self.yen_slider = ttk.Scale(yen_frame, from_=0, to=1, orient=tk.HORIZONTAL, 
                                  variable=self.yen_threshold, command=self.update_segmentation)
        self.yen_slider.pack(side=tk.TOP)

        ttk.Button(frame, text="Reconstrucción 3D", command=self.show_3d).pack(side=tk.TOP, pady=5)

        self.slider = ttk.Scale(frame, from_=0, to=0, orient=tk.HORIZONTAL, command=self.update_images)
        self.slider.pack(fill=tk.X, pady=5)

        self.fig, self.axes = plt.subplots(1, 3, figsize=(12, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_segmentation(self, event=None):
        if self.volume is None:
            return
            
        # Liberar memoria antes de la nueva segmentación
        gc.collect()
            
        self.seg_otsu = segment_image(self.volume, method="otsu", manual_threshold=self.otsu_threshold.get())
        self.seg_yen = segment_image(self.volume, method="yen", manual_threshold=self.yen_threshold.get())
        self.update_images()

    def load_dicom(self):
        folder = filedialog.askdirectory()
        if not folder:
            return

        # Liberar memoria antes de cargar nuevos datos
        gc.collect()

        volume, slices = load_dicom_series(folder)
        volume = preprocess_image(volume)

        self.volume = volume
        self.seg_otsu = segment_image(volume, method="otsu", manual_threshold=self.otsu_threshold.get())
        self.seg_yen = segment_image(volume, method="yen", manual_threshold=self.yen_threshold.get())

        self.slider.configure(to=volume.shape[2] - 1)
        self.index.set(volume.shape[2] // 2)
        self.slider.set(volume.shape[2] // 2)

        self.update_images()

    def update_images(self, event=None):
        if self.volume is None:
            return

        i = int(float(self.slider.get()))
        
        # Limpiar los ejes antes de actualizar
        for ax in self.axes:
            ax.clear()
            
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
    # Configurar el backend de matplotlib para mejor rendimiento
    plt.switch_backend('TkAgg')
    
    # Configurar el número de hilos de NumPy
    np.set_printoptions(threshold=np.inf)
    
    root = tk.Tk()
    app = DICOMApp(root)
    root.mainloop()
