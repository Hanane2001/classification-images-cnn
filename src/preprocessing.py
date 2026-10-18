import pandas as pd
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from exploration import PATH, SIZE_IMG, chemins_images_valides

MAPING_CLS = {
    "buildings": 0,
    "forest": 1,
    "glacier": 2,
    "mountain": 3,
    "sea": 4,
    "street": 5
}

tfkl = tf.keras.layers
flip = tfkl.RandomFlip(mode='horizontal')
rotate = tfkl.RandomRotation(factor=0.05)
zoom = tfkl.RandomZoom(height_factor=0.1, width_factor=0.1)
translate = tfkl.RandomTranslation(height_factor=0.05, width_factor=0.05)
contrast = tfkl.RandomContrast(factor=0.1)

def normaliser_images(image):
    image = tf.cast(image, tf.float32)
    new_img = image / 255.0 # ou just fait new_img = tf.image.per_image_standardization(image)
    return new_img

def image_process(image_path, image_size=(150, 150)):
    image_encode = tf.io.read_file(image_path)
    image_decode = tf.io.decode_jpeg(image_encode, channels=3)
    resized_image = tf.image.resize(image_decode, image_size)
    return normaliser_images(resized_image)

def augment_img(img):
    img = flip(img)
    img = rotate(img)
    img = zoom(img)
    img = translate(img)
    img = contrast(img)
    return tf.clip_by_value(img, 0.0, 1.0)


def chargement_par_batch(ds, batch_size=32, augment=False):
    if augment:
        ds = ds.shuffle(1000)
    prc_img = ds.map(lambda x, y: (image_process(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        prc_img = prc_img.map(lambda x, y: (augment_img(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    batch_img = prc_img.batch(batch_size)
    return batch_img.prefetch(tf.data.AUTOTUNE)

def data_x_y(path):
    images = chemins_images_valides(path)
    X = []
    y = []
    for chemin, classe, fichier in images:
        X.append(chemin)
        y.append(classe)
    return X, y
 

def separer_dataset(X, y):
    x_train_val, x_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=1/9, random_state=42, stratify=y_train_val)
    return x_train, x_val, x_test, y_train, y_val, y_test

def cree_data_tf(x_train, y_train, x_val, y_val, x_test, y_test):
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    return train_ds, validation_ds, test_ds

def pretret_charge_batch(train_ds, validation_ds, test_ds):
    train_ds = chargement_par_batch(train_ds, batch_size=32, augment=True)
    validation_ds = chargement_par_batch(validation_ds, batch_size=32, augment=False)
    test_ds = chargement_par_batch(test_ds, batch_size=32, augment=False)
    return train_ds, validation_ds, test_ds

def class_Indx(y_train, y_val, y_test):
    class_to_index = MAPING_CLS
    y_train = [class_to_index[classe] for classe in y_train]
    y_val = [class_to_index[classe] for classe in y_val]
    y_test = [class_to_index[classe] for classe in y_test]
    return y_train, y_val, y_test, class_to_index

def sauvegarder_separations(x_train, x_val, x_test, y_train, y_val, y_test, name):
    source = "data/processed"
    os.makedirs(source ,exist_ok=True)
    train = pd.DataFrame({"chemin_image": x_train, "label": y_train})
    val = pd.DataFrame({"chemin_image": x_val, "label": y_val})
    test = pd.DataFrame({"chemin_image": x_test, "label": y_test})
    train.to_csv(os.path.join(source, f"{name}_train.csv"), index=False)
    val.to_csv(os.path.join(source, f"{name}_val.csv"), index=False)
    test.to_csv(os.path.join(source, f"{name}_test.csv"), index=False)
    print("Les separations ont ete sauvgardees")

def visualiser_augmentation(nombre=5):
    images = chemins_images_valides(PATH)[:nombre]
    plt.figure(figsize=(12, 5))
    for i, (chemin, classe, fichier) in enumerate(images):
        image = Image.open(chemin).convert("RGB")
        image = image.resize(SIZE_IMG)
        image = np.array(image, dtype=np.float32) / 255.0
        image = tf.convert_to_tensor(image)
        img_aug = augment_img(image)
        plt.subplot(2, nombre, i+1)
        plt.imshow(image.numpy())
        plt.title(f"Originale\n {classe}")
        plt.axis("off")
        plt.subplot(2, nombre, nombre+i+1)
        plt.imshow(tf.clip_by_value(img_aug, 0, 1).numpy())
        plt.title(f"Augmentee\n {classe}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def main_preprocessing():
    print("Debut du preprocessing...")
    X, y = data_x_y(PATH)
    print(f"Nombre total d'images: {len(X)}")
    x_train, x_val, x_test, y_train, y_val, y_test = separer_dataset(X, y)
    print(f"Train: {len(x_train)} images")
    print(f"Validation: {len(x_val)} images")
    print(f"Test: {len(x_test)} images")
    sauvegarder_separations(x_train, x_val, x_test, y_train, y_val, y_test, name="images")
    y_train, y_val, y_test, class_to_index = class_Indx(y_train, y_val, y_test)
    train_ds, validation_ds, test_ds = cree_data_tf(x_train,y_train, x_val, y_val, x_test, y_test)
    train_ds, validation_ds, test_ds = pretret_charge_batch(train_ds, validation_ds, test_ds)
    return train_ds, validation_ds, test_ds, class_to_index

# if __name__ == "__main__":
#     visualiser_augmentation(nombre=5)
#     train_ds, validation_ds, test_ds, class_to_index = main_preprocessing()
