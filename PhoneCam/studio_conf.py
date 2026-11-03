"""Réglages du Studio, sans dépendance lourde : l'interface les lit dès le lancement."""

_VIDEOS = "*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.mts *.m2ts *.mxf *.wmv *.flv *.mpg *.mpeg *.ts *.3gp *.ogv"
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
_IMAGES = " ".join("*" + e for e in IMAGE_EXT)
MEDIA_FILTER = f"Vidéos et images ({_VIDEOS} {_IMAGES});;Vidéos ({_VIDEOS});;Images ({_IMAGES});;Tous les fichiers (*)"

TRACK_DEFAULTS = {
    "smooth": 60,        # lissage des repères sur la vidéo, %
    "det_size": 1024,    # taille d'analyse du détecteur
    "det_score": 35,     # confiance minimale, %
    "rescue": True,      # recherche ciblée quand un visage est perdu
    "gap": 0.75,         # trou comblé au plus (s)
    "min_len": 0.2,      # visage ignoré s'il dure moins (s)
}

# Réglages du moteur repris de la page Moteur de l'espace Direct.
ENGINE_DEFAULTS = {
    "engine": "plus",           # plus, ou classic : moteur d'origine de Deep-Live-Cam (inswapper)
    "occluder": "xseg_1",       # détecteur d'occlusion : xseg_1, xseg_2 ou xseg_3
    "occ_threshold": 50,        # XSeg : note sous laquelle un pixel compte comme caché, %
    "occ_islands": 0,           # XSeg : taches ignorées dans le visage, % de sa surface
    "occ_steady": 0,            # XSeg : masque stabilisé d'une image à l'autre, %
    "landmarker": "hrffa",      # repères du visage : hrffa ou detector
    "regions": ["skin", "brows", "eyes", "glasses", "nose", "lips", "mouth"],   # zones remplacées
    "expr_upper": True,         # expressions du haut du visage
    "expr_lower": True,         # expressions du bas du visage
    "opacity": 100,             # visage généré mêlé à l'image d'origine, %
    "sharpness": 0,             # netteté du visage, dixièmes
    "mirror": False,
    "color_correction": False,  # « Fix Blueish Cam » : rouge et bleu inversés à l'entrée
    "interpolation": False,     # lissage temporel
    "interpolation_force": 0,   # %
    "poisson_blend": False,     # moteur d'origine
    "mouth_mask_size": 0,       # moteur d'origine : 0 = coupé
    "mask_feather_ratio": 12,
    "mask_down_size": 10,       # %
    "mask_size": 100,           # %
}

# nom : (encodeur, extension, arguments, format de pixels)
CODECS = {
    "H.264 (NVENC, rapide)": ("h264", "mp4", ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr",
                                               "-b:v", "0", "-cq", "{q}", "-spatial-aq", "1", "-temporal-aq", "1"], "yuv420p"),
    "H.265 (NVENC, rapide)": ("hevc", "mp4", ["-c:v", "hevc_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr",
                                               "-b:v", "0", "-cq", "{q}", "-tag:v", "hvc1"], "p010le"),
    "AV1 (NVENC, compact)": ("av1", "mp4", ["-c:v", "av1_nvenc", "-preset", "p6", "-rc", "vbr", "-b:v", "0",
                                             "-cq", "{q}"], "p010le"),
    "H.264 (x264, processeur)": ("h264", "mp4", ["-c:v", "libx264", "-preset", "slow", "-crf", "{q}"], "yuv420p"),
    "H.265 (x265, processeur)": ("hevc", "mp4", ["-c:v", "libx265", "-preset", "medium", "-crf", "{q}",
                                                  "-tag:v", "hvc1"], "yuv420p10le"),
    "ProRes 422 HQ (montage)": ("prores", "mov", ["-c:v", "prores_ks", "-profile:v", "3", "-vendor", "apl0"], "yuv422p10le"),
}
QUALITIES = {"Maximale": 14, "Très haute": 18, "Haute": 22, "Standard": 26}
SIZES = {"Résolution d'origine": None, "2160p (4K)": 2160, "1440p": 1440, "1080p": 1080, "720p": 720}

LOOK_DEFAULTS = {
    "temperature": 0,      # −100 froid … 100 chaud
    "tint": 0,             # −100 vert … 100 magenta
    "exposure": 0,         # −100 … 100 (± 2 diaphragmes)
    "contrast": 0,
    "highlights": 0,
    "shadows": 0,
    "whites": 0,
    "blacks": 0,
    "saturation": 0,
    "vibrance": 0,
    "fade": 0,             # noirs relevés (aspect film), 0 … 100
    "teal_orange": 0,      # ombres bleu-vert, peaux orangées, 0 … 100
    "mono": 0,             # noir et blanc, 0 … 100
    "shadow_hue": 220, "shadow_tint": 0,        # teinte des ombres
    "mid_hue": 40, "mid_tint": 0,               # teinte des tons moyens
    "high_hue": 40, "high_tint": 0,             # teinte des hautes lumières
    "lut_path": "",        # table .cube importée
    "lut_mix": 100,
    "sharpen": 0,          # netteté, 0 … 100
    "bloom": 0,            # lueur autour des hautes lumières, 0 … 100
    "vignette": 0,         # −100 (éclaircit) … 100 (assombrit)
    "grain": 0,            # grain de film, 0 … 100
}

LOOK_PRESETS = {
    "Neutre": {},
    "Naturel lumineux": {"exposure": 6, "contrast": 8, "shadows": 12, "vibrance": 15, "sharpen": 15},
    "Cinéma": {"contrast": 18, "highlights": -18, "shadows": 8, "teal_orange": 35, "vibrance": 8,
               "vignette": 18, "grain": 10, "bloom": 8},
    "Blockbuster": {"contrast": 28, "blacks": -10, "teal_orange": 62, "saturation": 5, "vignette": 25,
                    "sharpen": 15, "bloom": 12},
    "Heure dorée": {"temperature": 32, "tint": 4, "contrast": 10, "shadows": 10, "vibrance": 18,
                    "high_hue": 40, "high_tint": 22, "bloom": 22},
    "Nuit bleue": {"temperature": -35, "exposure": -22, "contrast": 16, "saturation": -14,
                   "shadow_hue": 225, "shadow_tint": 32, "vignette": 30, "grain": 12},
    "Néon": {"contrast": 22, "saturation": 18, "shadow_hue": 285, "shadow_tint": 34, "high_hue": 195,
             "high_tint": 16, "bloom": 30},
    "Film argentique": {"contrast": -6, "fade": 18, "saturation": -10, "temperature": 8,
                        "shadow_hue": 200, "shadow_tint": 10, "high_hue": 45, "high_tint": 10, "grain": 32},
    "Années 70": {"temperature": 22, "fade": 26, "saturation": -16, "shadow_hue": 30, "shadow_tint": 14,
                  "grain": 24, "vignette": 16},
    "Froid et sobre": {"temperature": -14, "saturation": -24, "contrast": 12, "shadows": -6},
    "Noir et blanc": {"mono": 100, "contrast": 24, "grain": 18, "vignette": 14},
    "Ampoule chaude": {"temperature": 55, "exposure": -8, "contrast": 8, "vignette": 12},
    "Pièce sombre": {"exposure": -38, "contrast": 12, "temperature": 18, "vignette": 34, "grain": 18},
}
