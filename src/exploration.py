import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import hashlib
from PIL import Image

FORMAT_IMG = {".jpg", ".jpeg"}
CLASSE_CAT = {
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
}
Dossier_Name = Path("data/raw")
PATH = Dossier_Name / "seg_train"
SIZE_IMG = (150, 150)

def explorer_dataset(path):
    if not os.path.isdir(path):
        return "Ce chemin n'est pas un dossier"
    return os.listdir(path)

def compter_images_par_classe(path):
    list_cat = explorer_dataset(path)
    res = {}
    for cl in list_cat:
        ch = os.path.join(path, cl)
        if os.path.isdir(ch):
            res[cl] = len(os.listdir(ch))
    return res


def verifier_images(path):
    list_cat = explorer_dataset(path)
    res = {}
    for cl in list_cat:
        ch = os.path.join(path, cl)
        if os.path.isdir(ch):
            res[cl] = []
            for fichier in os.listdir(ch):
                chemin_image = os.path.join(ch, fichier)
                if os.path.isfile(chemin_image):
                    extension = os.path.splitext(fichier)[1].lower()
                    if extension in FORMAT_IMG:
                        try:
                            with Image.open(chemin_image) as image:
                                image.verify()
                        except Exception:
                            res[cl].append(fichier)
    return res

def chemins_images_valides(path):
    res = verifier_images(path)
    images = []
    for cl in res:
        ch = os.path.join(path, cl)
        for fichier in os.listdir(ch):
            if fichier in res[cl]:
                continue
            chemin = os.path.join(ch, fichier)
            if os.path.isfile(chemin):
                if os.path.splitext(fichier)[1].lower() in FORMAT_IMG:
                    images.append((chemin, cl, fichier))
    return images


def afficher_counte_images_par_classe(path):
    res = compter_images_par_classe(path)
    plt.figure(figsize=(9, 5))
    plt.bar(res.keys(), res.values())
    plt.title("Nombre d'images dans chaque classe")
    plt.xlabel("Classe")
    plt.ylabel("Nombre d'images")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def equilibre(df, tolerance=0.05):
    res = df.mean()
    ecart = abs(df - res) / res
    if (ecart <= tolerance).all():
        return True
    return False

def calcule_percent(df):
    percent = []
    total = df.sum()
    for i in df:
        per = (i / total) * 100
        percent.append(per)
    print(percent)
    return percent

def max_min_img(df):
    max_images = df.max()
    id_max = df.idxmax()
    print(f"{id_max}: {max_images}")
    min_images = df.min()
    id_min = df.idxmin()
    print(f"{id_min}: {min_images}")
    return max_images, id_max, min_images, id_min


def analyser_distribution_classes(path):
    res = compter_images_par_classe(path)
    df = pd.Series(res)
    max_images, id_max, min_images, id_min = max_min_img(df)
    percent = calcule_percent(df)
    if equilibre(df):
        print("data est equilibre")
    else:
        print("data n'est pas equilibre")
    return df, percent, max_images, id_max, min_images, id_min
    

def analyser_dimensions_images(path):
    for chemin, cl, fichier in chemins_images_valides(path):
        try:
            img = Image.open(chemin)
            if img.size != SIZE_IMG:
                print(f"size: {img.size}")
        except Exception:
            continue

def Affiche_partie_Images_par_classe(path, part=24):
    images = chemins_images_valides(path)
    for cl in CLASSE_CAT:
        selection = [img for img in images if img[1] == cl][:part]
        if not selection:
            continue
        plt.figure(figsize=(12, 8))
        for i, (chemin, classe, fichier) in enumerate(selection):
            with Image.open(chemin) as img:
                plt.subplot(4, 6, i + 1)
                plt.imshow(img.copy())
                plt.title(f"{classe} - {img.size}")
                plt.axis("off")
        plt.tight_layout()
        plt.show()

def hash_image(file_path, algorithm='sha256'):
    hasher = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for ch in iter(lambda: f.read(4096), b''):
            hasher.update(ch)
    return hasher.hexdigest()

def compare_hashes(hash_f, hash_e):
    if hash_f == hash_e:
        return True
    return False

def detecter_doublons(path):
    images = chemins_images_valides(path)
    hashes = {}
    doublons = []
    for chemin, cl, fichier in images:
        try:
            h = hash_image(chemin)
            if h in hashes:
                doublons.append((hashes[h], chemin))
            else:
                hashes[h] = chemin
        except Exception:
            continue
    return doublons


def delete_duplicate(path):
    db = detecter_doublons(path)
    for (i, j) in db:
        if os.path.exists(j):
            os.remove(j)
            print(f"Image doublon supprimee: {j}")
    return db


def main_exploration():
    print("Exploration du dataset...")
    print(explorer_dataset(PATH))
    print("\nNombre d'images par classe:")
    print(compter_images_par_classe(PATH))
    print("\nVerification des images...")
    img_corr = verifier_images(PATH)
    print(img_corr)
    analyser_distribution_classes(PATH)
    Affiche_partie_Images_par_classe(PATH)
    print("\nAnalyse des dimensions:")
    analyser_dimensions_images(PATH)
    print("\nDetection des doublons:")
    doublons = detecter_doublons(PATH)
    print(f"Nombre de paires de doublons: {len(doublons)}")
    print("\nSuppression des doublons:")
    delete_duplicate(PATH)
    

if __name__ == "__main__":
    main_exploration()
