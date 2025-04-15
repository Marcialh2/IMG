import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pydicom
from skimage.filters import threshold_otsu, threshold_yen, gaussian
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import skimage.measure
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import multiprocessing
import sys

# === Cargar datos DICOM ===
def load_single_dicom(file_path):
    try:
        return pydicom.dcmread(file_path)
    except Exception as e:
        print(f"Error al cargar {file_path}: {str(e)}")
        return None

def load_dicom_series(dicom_folder):
    try:
        dicom_files = sorted([f for f in os.listdir(dicom_folder) if f.endswith(".dcm")])
        if not dicom_files:
            raise FileNotFoundError("No se encontraron archivos DICOM en la carpeta especificada.")
        
        file_paths = [os.path.join(dicom_folder, f) for f in dicom_files]
        
        # Usar ThreadPoolExecutor para cargar archivos DICOM en paralelo
        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), len(file_paths))) as executor:
            slices = list(filter(None, executor.map(load_single_dicom, file_paths)))
        
        if not slices:
            raise ValueError("No se pudieron cargar archivos DICOM válidos.")
        
        slices.sort(key=lambda x: x.InstanceNumber)
        pixel_data = np.stack([s.pixel_array for s in slices], axis=-1)
        return pixel_data, slices
    except Exception as e:
        raise Exception(f"Error al cargar la serie DICOM: {str(e)}")

# === Preprocesamiento ===
def preprocess_slice(slice_data):
    try:
        # Usar operaciones in-place para reducir el uso de memoria
        slice_data = gaussian(slice_data, sigma=1)
        slice_min = np.min(slice_data)
        slice_max = np.max(slice_data)
        slice_data -= slice_min
        slice_data /= (slice_max - slice_min)
        return slice_data
    except Exception as e:
        print(f"Error en preprocesamiento: {str(e)}")
        return None

def preprocess_image(volume):
    try:
        # Procesar en lotes para mejorar el rendimiento
        batch_size = min(10, volume.shape[2])  # Procesar hasta 10 slices a la vez
        processed_slices = []
        
        for i in range(0, volume.shape[2], batch_size):
            batch_end = min(i + batch_size, volume.shape[2])
            batch = volume[:,:,i:batch_end]
            
            # Vectorizar el procesamiento del lote
            batch = gaussian(batch, sigma=1)
            batch_min = np.min(batch)
            batch_max = np.max(batch)
            batch = (batch - batch_min) / (batch_max - batch_min)
            
            processed_slices.extend([batch[:,:,j] for j in range(batch.shape[2])])
        
        return np.stack(processed_slices, axis=2)
    except Exception as e:
        raise Exception(f"Error en el preprocesamiento: {str(e)}")

# === Segmentación ===
def segment_slice(slice_data, method="otsu", custom_threshold=None):
    try:
        if custom_threshold is not None:
            threshold = custom_threshold
        elif method == "otsu":
            threshold = threshold_otsu(slice_data)
        elif method == "yen":
            threshold = threshold_yen(slice_data)
        else:
            raise ValueError("Método de segmentación no válido. Use 'otsu' o 'yen'.")
        
        # Usar operación in-place
        return slice_data > threshold
    except Exception as e:
        print(f"Error en segmentación: {str(e)}")
        return None

def segment_image(volume, method="otsu", custom_threshold=None):
    try:
        # Procesar en lotes para mejorar el rendimiento
        batch_size = min(10, volume.shape[2])
        segmented_slices = []
        
        # Imprimir información sobre el método seleccionado
        print(f"Segmentando con método: {method}")
        
        for i in range(0, volume.shape[2], batch_size):
            batch_end = min(i + batch_size, volume.shape[2])
            batch = volume[:,:,i:batch_end]
            
            # Procesar cada slice individualmente para calcular el umbral correctamente
            for j in range(batch.shape[2]):
                slice_data = batch[:,:,j]
                
                if custom_threshold is not None:
                    threshold = custom_threshold
                    print(f"Slice {i+j}: Usando umbral personalizado: {threshold}")
                elif method == "otsu":
                    threshold = threshold_otsu(slice_data)
                    print(f"Slice {i+j}: Umbral Otsu: {threshold}")
                elif method == "yen":
                    threshold = threshold_yen(slice_data)
                    print(f"Slice {i+j}: Umbral Yen: {threshold}")
                
                batch[:,:,j] = slice_data > threshold
            
            segmented_slices.extend([batch[:,:,j] for j in range(batch.shape[2])])
        
        return np.stack(segmented_slices, axis=2)
    except Exception as e:
        raise Exception(f"Error en la segmentación: {str(e)}")

# === Visualización 3D ===
def visualize_3d(volume):
    try:
        # Reducir la resolución para mejorar el rendimiento de la visualización
        scale_factor = 0.5
        small_volume = volume[::2, ::2, ::2]
        
        verts, faces, _, _ = skimage.measure.marching_cubes(small_volume, level=0)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        mesh = Poly3DCollection(verts[faces], alpha=0.7)
        mesh.set_facecolor((0.5, 0.5, 1))
        ax.add_collection3d(mesh)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(0, small_volume.shape[0])
        ax.set_ylim(0, small_volume.shape[1])
        ax.set_zlim(0, small_volume.shape[2])
        plt.tight_layout()
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"Error en la visualización 3D: {str(e)}")

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

        self.use_custom_threshold_otsu = tk.BooleanVar(value=False)
        self.custom_threshold_otsu = tk.DoubleVar(value=0.5)

        self.use_custom_threshold_yen = tk.BooleanVar(value=False)
        self.custom_threshold_yen = tk.DoubleVar(value=0.5)

        self.setup_ui()

    def setup_ui(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Button(frame, text="Cargar DICOM", command=self.load_dicom).pack(side=tk.TOP, pady=5)

        method_frame = ttk.Frame(frame)
        method_frame.pack(side=tk.TOP, pady=5)
        ttk.Label(method_frame, text="Método para 3D:").pack(side=tk.LEFT)
        method_selector = ttk.Combobox(method_frame, textvariable=self.selected_3d_method, state="readonly")
        method_selector['values'] = ["Otsu", "Yen"]
        method_selector.pack(side=tk.LEFT, padx=5)

        ttk.Button(frame, text="Reconstrucción 3D", command=self.show_3d).pack(side=tk.TOP, pady=5)

        # === Umbral Otsu personalizado ===
        otsu_frame = ttk.LabelFrame(frame, text="Umbral Otsu personalizado", padding=10)
        otsu_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(otsu_frame, text="Usar umbral Otsu personalizado",
                        variable=self.use_custom_threshold_otsu, command=self.update_segmentation).pack(anchor='w')
        otsu_slider_frame = ttk.Frame(otsu_frame)
        otsu_slider_frame.pack(fill=tk.X)
        ttk.Label(otsu_slider_frame, text="Umbral:").pack(side=tk.LEFT)
        ttk.Label(otsu_slider_frame, textvariable=self.custom_threshold_otsu).pack(side=tk.RIGHT)
        ttk.Scale(otsu_slider_frame, from_=0.0, to=1.0, variable=self.custom_threshold_otsu,
                  command=lambda e: self.update_segmentation(), orient=tk.HORIZONTAL).pack(fill=tk.X)

        # === Umbral Yen personalizado ===
        yen_frame = ttk.LabelFrame(frame, text="Umbral Yen personalizado", padding=10)
        yen_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(yen_frame, text="Usar umbral Yen personalizado",
                        variable=self.use_custom_threshold_yen, command=self.update_segmentation).pack(anchor='w')
        yen_slider_frame = ttk.Frame(yen_frame)
        yen_slider_frame.pack(fill=tk.X)
        ttk.Label(yen_slider_frame, text="Umbral:").pack(side=tk.LEFT)
        ttk.Label(yen_slider_frame, textvariable=self.custom_threshold_yen).pack(side=tk.RIGHT)
        ttk.Scale(yen_slider_frame, from_=0.0, to=1.0, variable=self.custom_threshold_yen,
                  command=lambda e: self.update_segmentation(), orient=tk.HORIZONTAL).pack(fill=tk.X)

        self.slider = ttk.Scale(frame, from_=0, to=0, orient=tk.HORIZONTAL, command=self.update_images)
        self.slider.pack(fill=tk.X, pady=5)

        self.fig, self.axes = plt.subplots(1, 3, figsize=(12, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_dicom(self):
        try:
            folder = filedialog.askdirectory()
            if not folder:
                return

            volume, slices = load_dicom_series(folder)
            volume = preprocess_image(volume)

            self.volume = volume
            self.update_segmentation()

            self.slider.configure(to=volume.shape[2] - 1)
            self.index.set(volume.shape[2] // 2)
            self.slider.set(volume.shape[2] // 2)

            self.update_images()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def update_segmentation(self):
        try:
            if self.volume is None:
                return

            print("Actualizando segmentación...")
            
            if self.use_custom_threshold_otsu.get():
                thresh_otsu = self.custom_threshold_otsu.get()
                print(f"Usando umbral personalizado para Otsu: {thresh_otsu}")
            else:
                thresh_otsu = None
                print("Usando umbral automático para Otsu")

            if self.use_custom_threshold_yen.get():
                thresh_yen = self.custom_threshold_yen.get()
                print(f"Usando umbral personalizado para Yen: {thresh_yen}")
            else:
                thresh_yen = None
                print("Usando umbral automático para Yen")

            print("Segmentando con método Otsu...")
            self.seg_otsu = segment_image(self.volume, method="otsu", custom_threshold=thresh_otsu)
            
            print("Segmentando con método Yen...")
            self.seg_yen = segment_image(self.volume, method="yen", custom_threshold=thresh_yen)
            
            print("Actualizando imágenes...")
            self.update_images()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def update_images(self, event=None):
        try:
            if self.volume is None:
                return

            i = int(float(self.slider.get()))
            print(f"Mostrando slice {i}")
            
            # Limpiar los ejes antes de mostrar nuevas imágenes
            for ax in self.axes:
                ax.clear()
            
            # Mostrar la imagen original
            self.axes[0].imshow(self.volume[:, :, i], cmap='gray')
            self.axes[0].set_title("Original")
            
            # Mostrar la imagen segmentada con Otsu
            if self.seg_otsu is not None:
                print(f"Mostrando segmentación Otsu para slice {i}")
                self.axes[1].imshow(self.seg_otsu[:, :, i], cmap='gray')
                self.axes[1].set_title("Otsu")
            else:
                print("No hay segmentación Otsu disponible")
                self.axes[1].set_title("Otsu (No disponible)")
            
            # Mostrar la imagen segmentada con Yen
            if self.seg_yen is not None:
                print(f"Mostrando segmentación Yen para slice {i}")
                self.axes[2].imshow(self.seg_yen[:, :, i], cmap='gray')
                self.axes[2].set_title("Yen")
            else:
                print("No hay segmentación Yen disponible")
                self.axes[2].set_title("Yen (No disponible)")

            for ax in self.axes:
                ax.axis('off')
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_3d(self):
        try:
            if self.selected_3d_method.get() == "Otsu" and self.seg_otsu is not None:
                visualize_3d(self.seg_otsu)
            elif self.selected_3d_method.get() == "Yen" and self.seg_yen is not None:
                visualize_3d(self.seg_yen)
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == '__main__':
    try:
        root = tk.Tk()
        app = DICOMApp(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Error", f"Error fatal: {str(e)}")
        sys.exit(1)
