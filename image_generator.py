"""
Script de génération de documents synthétiques (fond blanc + texte + graphique).

Étape actuelle : génère le "fond" du document (sans la mention C1/C2/C3,
qui sera ajoutée dans une prochaine étape).

Pour chaque document généré, le script produit :
- une image .jpg du document
- (préparation pour la suite : un fichier .txt d'annotation YOLO, une fois
  la mention C1/C2/C3 intégrée)
"""

import os
import random
from io import BytesIO

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sans interface graphique
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from faker import Faker

# ============================================================
# PARAMÈTRES GÉNÉRAUX — à ajuster selon tes besoins
# ============================================================

NB_DOCUMENTS = 10       # <-- Variable principale : nombre de documents à générer
DOSSIER_SORTIE = "dataset_inference/Image_class"   # dossier où seront sauvegardées les images

DPI = 150
MARGE_MIN_CM = 2.0
MARGE_MAX_CM = 3.0

# Liste de polices disponibles sur le système (à adapter/compléter avec tes propres .ttf)
POLICES_DISPONIBLES = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
]

fake = Faker("fr_FR")


# ============================================================
# ÉTAPE 1 — Conversion cm -> pixels et dimensions A4
# ============================================================

def cm_to_px(cm, dpi=DPI):
    return int(cm / 2.54 * dpi)


LARGEUR_A4 = cm_to_px(21)     # ~1240 px à 150 dpi
HAUTEUR_A4 = cm_to_px(29.7)   # ~1754 px à 150 dpi


# ============================================================
# ÉTAPE 2 — Génération des marges aléatoires + canevas blanc
# ============================================================

def generer_marges():
    marge_top = cm_to_px(random.uniform(MARGE_MIN_CM, MARGE_MAX_CM))
    marge_bottom = cm_to_px(random.uniform(MARGE_MIN_CM, MARGE_MAX_CM))
    marge_left = cm_to_px(random.uniform(MARGE_MIN_CM, MARGE_MAX_CM))
    marge_right = cm_to_px(random.uniform(MARGE_MIN_CM, MARGE_MAX_CM))
    return marge_top, marge_bottom, marge_left, marge_right


def creer_canevas():
    canevas = Image.new("RGB", (LARGEUR_A4, HAUTEUR_A4), color="white")

    marge_top, marge_bottom, marge_left, marge_right = generer_marges()

    zone_utile = {
        "x0": marge_left,
        "y0": marge_top,
        "x1": LARGEUR_A4 - marge_right,
        "y1": HAUTEUR_A4 - marge_bottom,
    }

    return canevas, zone_utile


# ============================================================
# ÉTAPE 3 — Génération du graphique (matplotlib)
# ============================================================

def generer_donnees_aleatoires():
    nb_categories = random.randint(3, 7)
    categories = [f"Cat {i + 1}" for i in range(nb_categories)]
    valeurs = np.random.randint(5, 100, size=nb_categories)
    return categories, valeurs


def generer_graphique():
    type_graphique = random.choice(["bar", "line", "pie"])
    categories, valeurs = generer_donnees_aleatoires()

    largeur_px, hauteur_px = 500, 350
    dpi_fig = 100
    fig, ax = plt.subplots(figsize=(largeur_px / dpi_fig, hauteur_px / dpi_fig), dpi=dpi_fig)

    couleurs = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    couleur_unique = random.choice(["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"])

    if type_graphique == "bar":
        ax.bar(categories, valeurs, color=couleur_unique)
    elif type_graphique == "line":
        ax.plot(categories, valeurs, marker="o", color=couleur_unique, linewidth=2)
    elif type_graphique == "pie":
        ax.pie(valeurs, labels=categories, colors=couleurs, autopct="%1.0f%%")

    if random.random() < 0.6 and type_graphique != "pie":
        ax.set_title(random.choice(["Résultats trimestriels", "Répartition", "Évolution", "Statistiques"]))
    if random.random() < 0.4 and type_graphique != "pie":
        ax.grid(axis="y", linestyle="--", alpha=0.5)
    if type_graphique != "pie":
        ax.tick_params(axis="x", rotation=random.choice([0, 30, 45]))

    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", facecolor="white")
    plt.close(fig)  # évite les fuites mémoire sur de grosses générations
    buffer.seek(0)

    image_graphique = Image.open(buffer).convert("RGB")
    return image_graphique


def inserer_graphique(canevas, zone_utile):
    # Environ 1 document sur 4 n'a pas de graphique, pour la diversité
    if random.random() < 0.25:
        return canevas, None

    graphique = generer_graphique()

    largeur_cible = int((zone_utile["x1"] - zone_utile["x0"]) * random.uniform(0.5, 0.8))
    ratio = largeur_cible / graphique.width
    hauteur_cible = int(graphique.height * ratio)
    graphique_redim = graphique.resize((largeur_cible, hauteur_cible))

    # Position verticale aléatoire (haut / milieu / bas de la zone utile), pour que
    # le texte puisse s'écouler avant ET après le graphique plutôt que de laisser un vide
    hauteur_disponible = (zone_utile["y1"] - zone_utile["y0"]) - hauteur_cible
    pos_x = zone_utile["x0"]
    pos_y = zone_utile["y0"] + int(random.uniform(0.15, 0.75) * hauteur_disponible)

    canevas.paste(graphique_redim, (pos_x, pos_y))
    bbox_graphique = (pos_x, pos_y, pos_x + largeur_cible, pos_y + hauteur_cible)
    return canevas, bbox_graphique


# ============================================================
# ÉTAPE 4 — Génération et dessin du texte
# ============================================================

def charger_police(taille, chemin_police=None):
    if chemin_police is None:
        chemin_police = random.choice(POLICES_DISPONIBLES)
    return ImageFont.truetype(chemin_police, taille)


# Tailles de police selon le type de bloc, pour créer une vraie hiérarchie visuelle
TAILLE_TITRE_PRINCIPAL = (28, 34)
TAILLE_SOUS_TITRE = (20, 24)
TAILLE_TEXTE_COURANT = (15, 18)

BANQUE_TITRES = [
    "Rapport d'activité", "Synthèse annuelle", "Présentation générale",
    "Analyse des résultats", "Contexte et enjeux", "Méthodologie",
    "Résultats et perspectives", "Bilan trimestriel", "Recommandations",
]
BANQUE_SOUS_TITRES = [
    "Contexte", "Objectifs", "Méthodologie", "Résultats", "Analyse",
    "Perspectives", "Points clés", "Synthèse", "Recommandations", "Limites",
]


def wrap_texte(draw, texte, police, largeur_max):
    mots = texte.split()
    lignes = []
    ligne_courante = ""

    for mot in mots:
        essai = f"{ligne_courante} {mot}".strip()
        bbox = draw.textbbox((0, 0), essai, font=police)
        largeur_essai = bbox[2] - bbox[0]

        if largeur_essai <= largeur_max:
            ligne_courante = essai
        else:
            if ligne_courante:
                lignes.append(ligne_courante)
            ligne_courante = mot

    if ligne_courante:
        lignes.append(ligne_courante)

    return lignes


def _saute_zone_interdite(y, interligne, zone_interdite):
    """Si la ligne à la position y chevauche la zone interdite (le graphique),
    renvoie la position y à laquelle reprendre juste après le graphique."""
    if not zone_interdite:
        return y
    ligne_y0, ligne_y1 = y, y + interligne
    chevauche = not (ligne_y1 < zone_interdite[1] or ligne_y0 > zone_interdite[3])
    if chevauche:
        return zone_interdite[3] + 25
    return y


def dessiner_bloc_titre(draw, texte, x, y, largeur_max, zone_interdite, zone_utile, niveau="titre"):
    """Dessine un titre ou sous-titre et renvoie la nouvelle position y."""
    if niveau == "titre":
        taille = random.randint(*TAILLE_TITRE_PRINCIPAL)
    else:
        taille = random.randint(*TAILLE_SOUS_TITRE)

    # Les titres utilisent de préférence une police différente du corps de texte
    police = charger_police(taille)
    interligne = int(taille * 1.3)

    y = _saute_zone_interdite(y, interligne, zone_interdite)
    if y + interligne > zone_utile["y1"]:
        return y, False  # plus de place

    draw.text((x, y), texte, font=police, fill="black")
    y += interligne + int(taille * 0.4)  # espace après le titre
    return y, True


def dessiner_paragraphe(draw, x, y, largeur_max, zone_interdite, zone_utile):
    """Dessine un paragraphe de texte courant et renvoie la nouvelle position y."""
    taille_police = random.randint(*TAILLE_TEXTE_COURANT)
    police = charger_police(taille_police)
    interligne = int(taille_police * 1.5)

    texte = fake.paragraph(nb_sentences=random.randint(3, 9))
    lignes = wrap_texte(ImageDraw.Draw(Image.new("RGB", (1, 1))), texte, police, largeur_max)

    for ligne in lignes:
        y = _saute_zone_interdite(y, interligne, zone_interdite)

        if y + interligne > zone_utile["y1"]:
            return y, False  # page pleine

        draw.text((x, y), ligne, font=police, fill="black")
        y += interligne

    y += int(interligne * 0.6)  # espace entre paragraphes
    return y, True


def dessiner_texte(canevas, zone_utile, zone_interdite=None):
    """
    Remplit la zone utile avec un titre principal, puis une alternance de
    sous-titres et de paragraphes, JUSQU'À CE QUE LA PAGE SOIT PLEINE
    (au lieu d'un nombre fixe de paragraphes) afin d'éviter les grands vides.
    """
    draw = ImageDraw.Draw(canevas)
    largeur_max = zone_utile["x1"] - zone_utile["x0"]
    x = zone_utile["x0"]
    y = zone_utile["y0"]

    # Titre principal du document (quasi toujours présent)
    if random.random() < 0.9:
        titre = random.choice(BANQUE_TITRES)
        y, place_ok = dessiner_bloc_titre(draw, titre, x, y, largeur_max, zone_interdite, zone_utile, niveau="titre")
        if not place_ok:
            return canevas

    # Boucle de remplissage : alterne sous-titres et paragraphes tant qu'il reste de la place.
    # dernier_bloc_est_titre empêche d'avoir deux titres collés sans texte entre les deux.
    marge_securite = 40  # px restants sous lesquels on arrête de remplir
    dernier_bloc_est_titre = True  # après le titre principal, on force un paragraphe d'abord

    while y + marge_securite < zone_utile["y1"]:
        # Environ 1 bloc sur 3 est un sous-titre, sauf si le bloc précédent était déjà un titre
        if not dernier_bloc_est_titre and random.random() < 0.3:
            sous_titre = random.choice(BANQUE_SOUS_TITRES)
            y, place_ok = dessiner_bloc_titre(
                draw, sous_titre, x, y, largeur_max, zone_interdite, zone_utile, niveau="sous_titre"
            )
            dernier_bloc_est_titre = True
        else:
            y, place_ok = dessiner_paragraphe(draw, x, y, largeur_max, zone_interdite, zone_utile)
            dernier_bloc_est_titre = False

        if not place_ok:
            break

    return canevas


# ============================================================
# ÉTAPE 5 — Génération d'un document complet
# ============================================================

def generer_un_document():
    canevas, zone_utile = creer_canevas()
    canevas, bbox_graphique = inserer_graphique(canevas, zone_utile)
    canevas = dessiner_texte(canevas, zone_utile, zone_interdite=bbox_graphique)
    return canevas


# ============================================================
# ÉTAPE 6 — Boucle principale de génération
# ============================================================

def main():
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)

    for i in range(NB_DOCUMENTS):
        document = generer_un_document()

        nom_fichier = f"document_{i:05d}.jpg"
        chemin_fichier = os.path.join(DOSSIER_SORTIE, nom_fichier)

        # Qualité JPEG variable pour un peu de réalisme (à enrichir avec bruit/flou plus tard)
        qualite = random.randint(75, 95)
        document.save(chemin_fichier, "JPEG", quality=qualite)

        if (i + 1) % 10 == 0 or (i + 1) == NB_DOCUMENTS:
            print(f"{i + 1}/{NB_DOCUMENTS} documents générés")

    print(f"\nTerminé. {NB_DOCUMENTS} documents sauvegardés dans '{DOSSIER_SORTIE}/'")


if __name__ == "__main__":
    main()