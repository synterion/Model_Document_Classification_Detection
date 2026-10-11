"""
Script d'incrustation des mentions de classification (C1/C2/C3) sur des
documents déjà générés.

Entrée  : un dossier contenant des images de documents (.jpg/.png)
Sortie  : - un dossier d'images avec la mention incrustée
          - un dossier de fichiers .txt d'annotation au format YOLO
          - un fichier data.yaml prêt pour l'entraînement Ultralytics

Usage :
    Modifie les variables DOSSIER_IMAGES_ENTREE / DOSSIER_SORTIE ci-dessous,
    puis lance : python incruster_mentions.py
"""

import os
import random
import glob

from PIL import Image, ImageDraw, ImageFont

# ============================================================
# PARAMÈTRES GÉNÉRAUX
# ============================================================

DOSSIER_IMAGES_ENTREE = "dataset_inference/Image_class"   # dossier généré par generer_documents.py
DOSSIER_SORTIE = "dataset_inference"               # dossier de sortie (images + labels)

# Les 3 classes à incruster : (id_classe, texte_affiché)
# L'id_classe est celui qui sera écrit dans les fichiers d'annotation YOLO
MENTIONS = [
    (0, "C1 - Usage Interne"),
    (1, "C2 - Diffusion Restreinte"),
    (2, "C3 - Secret"),
]
NOMS_CLASSES = [texte for _, texte in MENTIONS]  # pour le data.yaml

# Polices utilisables pour la mention (idéalement différentes du corps de texte)
POLICES_MENTIONS = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

# Plage de tailles de police pour la mention (en pixels)
TAILLE_MIN = 22
TAILLE_MAX = 60

# Palette de couleurs pour la mention (RGB) — teintes réalistes de tampons/mentions administratives
COULEURS = [
    (200, 0, 0),      # rouge
    (0, 0, 0),        # noir
    (0, 60, 150),     # bleu
    (120, 0, 0),      # rouge foncé
    (90, 90, 90),     # gris foncé
]

# Plage de rotation (degrés). 0 = horizontal, valeurs élevées = façon tampon en diagonale
ROTATION_MIN = -35
ROTATION_MAX = 35

# Plage d'opacité (255 = opaque, valeurs plus basses = effet filigrane)
OPACITE_MIN = 140
OPACITE_MAX = 255

# Bandes verticales (en proportion de la hauteur totale) où peut être placée la mention.
# Une seule mention est incrustée par document ; sa zone verticale (haut / bas / libre)
# est tirée aléatoirement pour chaque document.
ZONES_PLACEMENT = {
    "haut": 0.35,    # bande du haut, façon en-tête
    "bas": 0.35,     # bande du bas, façon pied de page
    "libre": 0.30,   # n'importe où sur la page
}
ZONE_HAUT = (0.0, 0.15)   # les 15% supérieurs de la page
ZONE_BAS = (0.85, 1.0)    # les 15% inférieurs de la page
ZONE_LIBRE = (0.0, 1.0)   # toute la page


def tirer_bande_verticale():
    """Tire aléatoirement l'une des 3 zones de placement et renvoie son nom + ses bornes (y_min, y_max)."""
    zone = random.choices(
        population=list(ZONES_PLACEMENT.keys()),
        weights=list(ZONES_PLACEMENT.values()),
        k=1,
    )[0]
    bornes = {"haut": ZONE_HAUT, "bas": ZONE_BAS, "libre": ZONE_LIBRE}[zone]
    return zone, bornes


# Probabilité d'ajouter un cadre autour du texte (effet "tampon encadré")
PROBA_CADRE = 0.4


# ============================================================
# ÉTAPE 1 — Génération du calque de la mention (texte + rotation)
# ============================================================

def creer_calque_mention(texte, taille_image, autoriser_rotation=True):
    """
    Dessine le texte de la mention sur un calque transparent, avec une police,
    taille, couleur et opacité aléatoires.
    Si autoriser_rotation=True, le calque est pivoté d'un angle aléatoire
    (comportement pour la zone "libre"). Sinon le texte reste bien droit
    (comportement pour les zones "haut" et "bas").
    Renvoie le calque (RGBA) prêt à être collé sur le document.
    """
    taille_police = random.randint(TAILLE_MIN, TAILLE_MAX)
    chemin_police = random.choice(POLICES_MENTIONS)
    police = ImageFont.truetype(chemin_police, taille_police)

    couleur = random.choice(COULEURS)
    opacite = random.randint(OPACITE_MIN, OPACITE_MAX)
    couleur_rgba = couleur + (opacite,)

    # Calque temporaire juste pour mesurer la taille du texte
    calque_mesure = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    draw_mesure = ImageDraw.Draw(calque_mesure)
    bbox_texte = draw_mesure.textbbox((0, 0), texte, font=police)
    largeur_texte = bbox_texte[2] - bbox_texte[0]
    hauteur_texte = bbox_texte[3] - bbox_texte[1]

    # Marge autour du texte, utile si on ajoute un cadre
    marge = int(taille_police * 0.35)
    largeur_calque = largeur_texte + marge * 2
    hauteur_calque = hauteur_texte + marge * 2

    calque = Image.new("RGBA", (largeur_calque, hauteur_calque), (0, 0, 0, 0))
    draw = ImageDraw.Draw(calque)

    # Décalage pour compenser l'offset renvoyé par textbbox
    draw.text((marge - bbox_texte[0], marge - bbox_texte[1]), texte, font=police, fill=couleur_rgba)

    # Cadre autour du texte façon tampon, une fois sur deux environ
    if random.random() < PROBA_CADRE:
        epaisseur = max(2, taille_police // 15)
        draw.rectangle(
            [0, 0, largeur_calque - 1, hauteur_calque - 1],
            outline=couleur_rgba,
            width=epaisseur,
        )

    # Rotation du calque (expand=True agrandit le calque pour ne rien couper).
    # Seule la zone "libre" applique une rotation ; les zones "haut"/"bas" restent droites.
    if autoriser_rotation:
        angle = random.uniform(ROTATION_MIN, ROTATION_MAX)
        calque = calque.rotate(angle, expand=True, resample=Image.BICUBIC)

    return calque


# ============================================================
# ÉTAPE 2 — Collage sur le document + calcul de la bounding box
# ============================================================

def incruster_mention(image_document, texte_mention):
    """
    Colle UNE SEULE mention sur l'image du document.
    - Zone "libre" : position (x, y) aléatoire sur toute la page, avec rotation aléatoire.
    - Zones "haut"/"bas" : mention centrée horizontalement, sans rotation (texte droit),
      à l'intérieur de la bande verticale correspondante.
    Renvoie l'image modifiée + la bounding box (x0, y0, x1, y1) en pixels.
    """
    image_document = image_document.convert("RGB")
    largeur_doc, hauteur_doc = image_document.size

    nom_zone, zone_verticale = tirer_bande_verticale()
    autoriser_rotation = (nom_zone == "libre")

    calque = creer_calque_mention(texte_mention, image_document.size, autoriser_rotation=autoriser_rotation)
    largeur_calque, hauteur_calque = calque.size

    # Bornes en pixels de la bande verticale tirée pour cette mention
    y_min = int(zone_verticale[0] * hauteur_doc)
    y_max = int(zone_verticale[1] * hauteur_doc)
    hauteur_bande = max(1, y_max - y_min)

    # Si le calque est plus grand que la bande disponible (ou plus large que le
    # document), on le redimensionne pour qu'il tienne
    ratio = min(largeur_doc / largeur_calque, hauteur_bande / hauteur_calque, 1.0) * 0.95
    if ratio < 1.0:
        nouvelle_taille = (max(1, int(largeur_calque * ratio)), max(1, int(hauteur_calque * ratio)))
        calque = calque.resize(nouvelle_taille, Image.BICUBIC)
        largeur_calque, hauteur_calque = calque.size

    if nom_zone == "libre":
        # Position aléatoire en x sur toute la largeur, en y sur toute la page
        pos_x = random.randint(0, max(0, largeur_doc - largeur_calque))
        pos_y = random.randint(y_min, max(y_min, y_max - hauteur_calque))
    else:
        # Zones "haut"/"bas" : centrage horizontal, position verticale aléatoire
        # à l'intérieur de la bande (mais toujours centrée en x)
        pos_x = (largeur_doc - largeur_calque) // 2
        pos_y = random.randint(y_min, max(y_min, y_max - hauteur_calque))

    image_document.paste(calque, (pos_x, pos_y), calque)  # calque sert aussi de masque (alpha)

    # Bounding box précise : pixels réellement non-transparents du calque
    bbox_locale = calque.getchannel("A").getbbox()
    if bbox_locale is None:
        bbox_locale = (0, 0, largeur_calque, hauteur_calque)

    x0 = pos_x + bbox_locale[0]
    y0 = pos_y + bbox_locale[1]
    x1 = pos_x + bbox_locale[2]
    y1 = pos_y + bbox_locale[3]

    return image_document, (x0, y0, x1, y1)


# ============================================================
# ÉTAPE 3 — Conversion bounding box -> format YOLO normalisé
# ============================================================

def bbox_vers_yolo(bbox, largeur_image, hauteur_image):
    x0, y0, x1, y1 = bbox
    x_centre = ((x0 + x1) / 2) / largeur_image
    y_centre = ((y0 + y1) / 2) / hauteur_image
    largeur = (x1 - x0) / largeur_image
    hauteur = (y1 - y0) / hauteur_image
    return x_centre, y_centre, largeur, hauteur


# ============================================================
# ÉTAPE 4 — Traitement d'une image et écriture des fichiers de sortie
# ============================================================

def traiter_une_image(chemin_image, id_classe, texte_mention, dossier_sortie_images, dossier_sortie_labels):
    image = Image.open(chemin_image)
    image_annotee, bbox = incruster_mention(image, texte_mention)

    nom_base = os.path.splitext(os.path.basename(chemin_image))[0]

    # Sauvegarde de l'image annotée
    chemin_image_sortie = os.path.join(dossier_sortie_images, f"{nom_base}.jpg")
    image_annotee.save(chemin_image_sortie, "JPEG", quality=random.randint(80, 95))

    # Sauvegarde de l'annotation YOLO : une seule ligne, une seule mention par document
    x_centre, y_centre, largeur, hauteur = bbox_vers_yolo(bbox, *image_annotee.size)
    chemin_label_sortie = os.path.join(dossier_sortie_labels, f"{nom_base}.txt")
    with open(chemin_label_sortie, "w") as f:
        f.write(f"{id_classe} {x_centre:.6f} {y_centre:.6f} {largeur:.6f} {hauteur:.6f}\n")


# ============================================================
# ÉTAPE 5 — Écriture du fichier data.yaml (config Ultralytics)
# ============================================================

def ecrire_data_yaml(dossier_sortie):
    contenu = (
        f"path: {os.path.abspath(dossier_sortie)}\n"
        f"train: images\n"
        f"val: images\n"
        f"test: images\n"
        f"\n"
        f"nc: {len(NOMS_CLASSES)}\n"
        f"names: {NOMS_CLASSES}\n"
    )
    with open(os.path.join(dossier_sortie, "data.yaml"), "w") as f:
        f.write(contenu)


# ============================================================
# ÉTAPE 6 — Boucle principale
# ============================================================

def main():
    dossier_sortie_images = os.path.join(DOSSIER_SORTIE, "images")
    dossier_sortie_labels = os.path.join(DOSSIER_SORTIE, "labels")
    os.makedirs(dossier_sortie_images, exist_ok=True)
    os.makedirs(dossier_sortie_labels, exist_ok=True)

    chemins_images = sorted(
        glob.glob(os.path.join(DOSSIER_IMAGES_ENTREE, "*.jpg"))
        + glob.glob(os.path.join(DOSSIER_IMAGES_ENTREE, "*.png"))
    )

    if not chemins_images:
        print(f"Aucune image trouvée dans '{DOSSIER_IMAGES_ENTREE}'. Vérifie le chemin.")
        return

    random.shuffle(chemins_images)

    # Répartition équilibrée des 3 classes sur l'ensemble des images
    # (plutôt qu'un tirage purement aléatoire qui pourrait déséquilibrer les classes)
    assignations = [MENTIONS[i % len(MENTIONS)] for i in range(len(chemins_images))]
    random.shuffle(assignations)

    compteur_par_classe = {texte: 0 for _, texte in MENTIONS}

    for i, (chemin_image, (id_classe, texte_mention)) in enumerate(zip(chemins_images, assignations)):
        traiter_une_image(chemin_image, id_classe, texte_mention, dossier_sortie_images, dossier_sortie_labels)
        compteur_par_classe[texte_mention] += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(chemins_images):
            print(f"{i + 1}/{len(chemins_images)} images traitées")

    ecrire_data_yaml(DOSSIER_SORTIE)

    print(f"\nTerminé. Résultats dans '{DOSSIER_SORTIE}/'")
    print("Répartition par classe :")
    for texte, nb in compteur_par_classe.items():
        print(f"  {texte} : {nb} images")


if __name__ == "__main__":
    main()