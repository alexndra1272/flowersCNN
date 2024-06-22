# Este archivo sirve para mandar datos de train, test y validation a su respectiva carpeta

import os

import random
from shutil import copyfile
from sklearn.model_selection import train_test_split


original_path = "dataset/Data"

# Datos divididos en train, test y validation
base_dir = "dataset/Flowers"
os.makedirs(base_dir, exist_ok=True)


# Clases
classes = os.listdir(original_path)

train_dir = os.path.join(base_dir, "Train")
test_dir = os.path.join(base_dir, "Test")
val_dir = os.path.join(base_dir, "Validation")


for directory in [train_dir, test_dir, val_dir]:
    os.makedirs(directory, exist_ok=True)
    for class_name in classes:
        os.makedirs(os.path.join(directory, class_name), exist_ok=True)

# Recorrer cada clase y distribuir las imágenes
for class_name in classes:
    class_dir = os.path.join(original_path, class_name)
    all_images = os.listdir(class_dir)
    
    # Dividir en train y temp (que luego se dividirá en test y validation)
    train_images, temp_images = train_test_split(all_images, test_size=0.2, random_state=42)
    test_images, val_images = train_test_split(temp_images, test_size=0.5, random_state=42)
    
    # Función para copiar imágenes a sus respectivos directorios
    def copy_images(images, src_dir, dst_dir):
        for img in images:
            src = os.path.join(src_dir, img)
            dst = os.path.join(dst_dir, class_name, img)
            copyfile(src, dst)
    
    # Copiar imágenes a sus respectivos directorios de train, test y validation
    copy_images(train_images, class_dir, train_dir)
    copy_images(test_images, class_dir, test_dir)
    copy_images(val_images, class_dir, val_dir)

print("Dataset dividido exitosamente en train, test y validation.")
