"""Modèles du moteur « Plus » : téléchargement (versions officielles de FaceFusion, vérifiées par
CRC32) dans data\\models-plus, et préparation de l'analyseur de visage.
Supprimer data\\models-plus retire tout (moteurs TensorRT compris, dans trt-cache).

En ligne de commande : python models_plus.py  → télécharge ce qui manque.
"""
import os
import urllib.request
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "data", "models-plus")
BASE = "https://github.com/facefusion/facefusion-assets/releases/download/{tag}/{name}"
MODELS = [  # (version, nom, taille approximative en Mo)
    ("models-3.3.0", "hyperswap_1a_256", 384), ("models-3.3.0", "hyperswap_1b_256", 384),
    ("models-3.3.0", "hyperswap_1c_256", 384),
    ("models-3.1.0", "xseg_1", 67), ("models-3.0.0", "bisenet_resnet_34", 89),
    ("models-3.0.0", "live_portrait_feature_extractor", 3),
    ("models-3.0.0", "live_portrait_motion_extractor", 107),
    ("models-3.0.0", "live_portrait_generator", 212),
    ("models-3.5.0", "modnet", 25),   # silhouette de la personne : protège le décor des filtres de forme
]
# Téléchargés seulement si on les choisit (réglages avancés du moteur Plus).
OPTIONAL = {"xseg_2": ("models-3.1.0", 67), "xseg_3": ("models-3.2.0", 67), "hrffa": ("models-3.9.0", 36)}
# Classes de BiSeNet gardées : peau, sourcils, yeux, lunettes, nez, intérieur de la bouche, lèvres.
FACE_CLASSES = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
PARSER = "bisenet_face_256"
PARSER_VERSION = "argmax-hair-inner-5"   # change → le modèle dérivé est refait
HAIR_CLASS = 17
MOUTH_CLASSES = [11]   # intérieur de la bouche (dents, langue)


def _crc(path):
    c = 0
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            c = zlib.crc32(b, c)
    return format(c, "08x")


def missing():
    return [n for _, n, _ in MODELS if not os.path.isfile(os.path.join(MODELS_DIR, n + ".onnx"))]


def _fetch(tag, name, mb, log):
    path = os.path.join(MODELS_DIR, name + ".onnx")
    expected = urllib.request.urlopen(BASE.format(tag=tag, name=name + ".hash"), timeout=30).read().decode().strip()
    log(f"Téléchargement de {name} (~{mb} Mo)…")
    tmp = path + ".part"
    urllib.request.urlretrieve(BASE.format(tag=tag, name=name + ".onnx"), tmp)
    if _crc(tmp) != expected:
        os.remove(tmp)
        raise RuntimeError(f"{name} : fichier corrompu (CRC32), réessayer.")
    os.replace(tmp, path)


def download_optional(name, log=print):
    """Modèle facultatif (voir OPTIONAL), s'il manque."""
    if not os.path.isfile(os.path.join(MODELS_DIR, name + ".onnx")):
        os.makedirs(MODELS_DIR, exist_ok=True)
        tag, mb = OPTIONAL[name]
        _fetch(tag, name, mb, log)


def download(log=print):
    os.makedirs(MODELS_DIR, exist_ok=True)
    for tag, name, mb in MODELS:
        path = os.path.join(MODELS_DIR, name + ".onnx")
        if os.path.isfile(path):
            continue
        _fetch(tag, name, mb, log)
    prepare_parser(log)
    log("Modèles prêts.")


def prepare_parser(log=print):
    """bisenet_resnet_34 → bisenet_face_256 : sortie directe en 256 × 256, calculée sur le GPU,
    canal 0 « visage oui/non », canal 1 « cheveux / poils oui/non » (pour la barbe), canal 2
    « intérieur de la bouche oui/non » (dents, langue). Classe la
    plus probable parmi les 19, comme FaceFusion (une somme de probabilités donnait une peau à
    ~0,7 seulement, et le vrai visage transparaissait)."""
    out_path = os.path.join(MODELS_DIR, PARSER + ".onnx")
    stamp = out_path + ".version"
    if os.path.isfile(out_path) and os.path.isfile(stamp) and open(stamp).read() == PARSER_VERSION:
        return
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(os.path.join(MODELS_DIR, "bisenet_resnet_34.onnx"))
    g = m.graph
    opset = max(o.version for o in m.opset_import if o.domain in ("", "ai.onnx"))
    logits = g.output[0].name
    others = [c for c in range(19) if c not in FACE_CLASSES]
    not_hair = [c for c in range(19) if c != HAIR_CLASS]
    not_mouth = [c for c in range(19) if c not in MOUTH_CLASSES]
    g.initializer.extend([
        numpy_helper.from_array(np.array(FACE_CLASSES, np.int64), "pc_face"),
        numpy_helper.from_array(np.array(others, np.int64), "pc_other"),
        numpy_helper.from_array(np.array([HAIR_CLASS], np.int64), "pc_hair"),
        numpy_helper.from_array(np.array(not_hair, np.int64), "pc_nothair"),
        numpy_helper.from_array(np.array(MOUTH_CLASSES, np.int64), "pc_mouth"),
        numpy_helper.from_array(np.array(not_mouth, np.int64), "pc_notmouth"),
        numpy_helper.from_array(np.array([1], np.int64), "pc_axes"),
        numpy_helper.from_array(np.array([1, 1, 0.5, 0.5], np.float32), "pc_scales"),
        numpy_helper.from_array(np.array([], np.float32), "pc_roi"),
    ])

    def reduce_max(x, y):
        if opset >= 18:
            return helper.make_node("ReduceMax", [x, "pc_axes"], [y], keepdims=1)
        return helper.make_node("ReduceMax", [x], [y], axes=[1], keepdims=1)

    g.node.extend([
        helper.make_node("Gather", [logits, "pc_face"], ["pc_f"], axis=1), reduce_max("pc_f", "pc_fmax"),
        helper.make_node("Gather", [logits, "pc_other"], ["pc_o"], axis=1), reduce_max("pc_o", "pc_omax"),
        helper.make_node("Greater", ["pc_fmax", "pc_omax"], ["pc_is"]),
        helper.make_node("Cast", ["pc_is"], ["pc_isf"], to=TensorProto.FLOAT),
        helper.make_node("Gather", [logits, "pc_hair"], ["pc_h"], axis=1),
        helper.make_node("Gather", [logits, "pc_nothair"], ["pc_nh"], axis=1), reduce_max("pc_nh", "pc_nhmax"),
        helper.make_node("Greater", ["pc_h", "pc_nhmax"], ["pc_ish"]),
        helper.make_node("Cast", ["pc_ish"], ["pc_ishf"], to=TensorProto.FLOAT),
        helper.make_node("Gather", [logits, "pc_mouth"], ["pc_m"], axis=1), reduce_max("pc_m", "pc_mmax"),
        helper.make_node("Gather", [logits, "pc_notmouth"], ["pc_nm"], axis=1), reduce_max("pc_nm", "pc_nmmax"),
        helper.make_node("Greater", ["pc_mmax", "pc_nmmax"], ["pc_ism"]),
        helper.make_node("Cast", ["pc_ism"], ["pc_ismf"], to=TensorProto.FLOAT),
        helper.make_node("Concat", ["pc_isf", "pc_ishf", "pc_ismf"], ["pc_both"], axis=1),
        helper.make_node("Resize", ["pc_both", "pc_roi", "pc_scales"], ["face"], mode="linear"),
    ])
    while len(g.output):
        g.output.pop()
    g.output.append(helper.make_tensor_value_info("face", TensorProto.FLOAT, [1, 3, 256, 256]))
    onnx.checker.check_model(m)
    onnx.save(m, out_path)
    with open(stamp, "w") as f:
        f.write(PARSER_VERSION)
    log("Analyseur de visage préparé.")


def prepare_labels(log=print):
    """bisenet_resnet_34 → bisenet_labels_256 : classe la plus probable de chaque pixel (0 à 18),
    en 256 × 256. Sert seulement quand on choisit les zones du visage remplacées (réglages avancés) ;
    fichier séparé, l'analyseur habituel (bisenet_face_256) reste tel quel."""
    out_path = os.path.join(MODELS_DIR, "bisenet_labels_256.onnx")
    if os.path.isfile(out_path):
        return
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(os.path.join(MODELS_DIR, "bisenet_resnet_34.onnx"))
    g = m.graph
    logits = g.output[0].name
    g.initializer.extend([
        numpy_helper.from_array(np.array([1, 1, 0.5, 0.5], np.float32), "pl_scales"),
        numpy_helper.from_array(np.array([], np.float32), "pl_roi"),
    ])
    g.node.extend([
        helper.make_node("ArgMax", [logits], ["pl_arg"], axis=1, keepdims=1),
        helper.make_node("Cast", ["pl_arg"], ["pl_argf"], to=TensorProto.FLOAT),
        helper.make_node("Resize", ["pl_argf", "pl_roi", "pl_scales"], ["label"], mode="nearest"),
    ])
    while len(g.output):
        g.output.pop()
    g.output.append(helper.make_tensor_value_info("label", TensorProto.FLOAT, [1, 1, 256, 256]))
    onnx.checker.check_model(m)
    onnx.save(m, out_path + ".part")
    os.replace(out_path + ".part", out_path)
    log("Analyseur par zones préparé.")


if __name__ == "__main__":
    download()
