# PhoneCam

Remplacement de visage et changeur de voix en direct, de la webcam du PC ou d'un téléphone Android
vers une caméra virtuelle et un micro virtuel, et un Studio pour retoucher des vidéos, des images et
des sons après coup.

## Le moteur : FaceFusion

Le moteur **Plus** de PhoneCam est construit sur les modèles de [FaceFusion](https://github.com/facefusion/facefusion),
nettement plus réalistes que le collage d'origine de Deep-Live-Cam :

- **HyperSwap 1a / 1b / 1c** (256 px) : le remplacement du visage, bien plus naturel qu'inswapper ;
- **HRFFA** : 68 repères calculés sur toute la tête, qui suivent le visage jusqu'au profil
  complet et quand la tête bouge vite ;
- **LivePortrait** : les vraies expressions (bouche, sourire, sourcils, regard) réappliquées au
  visage généré ;
- **XSeg** : une main, un micro, un verre restent devant le visage ;
- **BiSeNet** : collage sur le visage seul, sans cheveux ni décor, zones au choix ;
- **MODNet** : détourage de la personne pour les retouches de silhouette ;
- les réglages avancés de FaceFusion : pixel boost, poids de l'identité, zones du masque,
  zones des expressions.

Par-dessus : stabilisation du cadrage de profil, garde anti-fuite (le vrai visage n'est jamais
montré quand la détection décroche), retouches façon Instagram / TikTok, rendu « vraie webcam »,
micro de téléphone simulé. Tout tourne sur la carte graphique avec TensorRT.

**Réglage conseillé** : page Moteur → **Moteur Plus** activé, modèle **HyperSwap 1a**, repères
**HRFFA**, lissage temporel coupé, anti-fuite actif avec « Ensuite » sur « Rester figé ».

[Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) reste en dessous pour la détection des
visages (InsightFace), le modèle inswapper 128 et les améliorateurs GFPGAN / GPEN.

## La voix

- **Direct** : ta voix transformée en temps réel par un modèle RVC (v1 / v2), avec **Smart** (seule
  la voix entre dans le modèle : clavier, souffle, ventilateur restent dehors), voix vivante,
  effets de studio (égaliseur, compresseur, de-esser) et réalisme d'un micro de smartphone.
- **Studio** : un fichier audio, un enregistrement au micro ou le son d'une vidéo, transformé avec
  une voix et les mêmes effets, à écouter en A/B puis à exporter (WAV, FLAC, MP3, M4A, ou la vidéo
  avec la nouvelle voix).
- **Deux voix fournies** (dossier `Voix`), prêtes dès l'installation ; on peut y ranger les siennes
  (un sous-dossier par voix, avec son `.pth` et son `.index`).

## Il faut

- Windows 10 ou 11 (64 bits)
- Une carte graphique NVIDIA récente (RTX conseillée), pilote à jour
- Environ 20 Go libres et Internet pour l'installation (environ 8,5 Go téléchargés)

Tout le reste est installé par `Installer.bat` : Python 3.12, les paquets Python, les modèles
(visage et voix), OBS Studio (pour sa caméra virtuelle), ffmpeg et adb.

## Installer

1. En haut de cette page : **Code → Download ZIP**, puis extraire le zip, par exemple dans Documents.
2. Double-clic sur **`Installer.bat`**. Si Windows affiche « Windows a protégé votre ordinateur »
   (fichier venu d'Internet) : **Informations complémentaires → Exécuter quand même**. Compter 20 à 40 minutes selon la connexion.
   Windows peut demander l'autorisation d'installer OBS. À la fin, l'installateur affiche la carte
   graphique trouvée. En cas d'erreur, le relancer reprend où il s'est arrêté.
3. Si OBS vient d'être installé : redémarrer le PC une fois.

## Lancer

Double-clic sur **`Lancer PhoneCam.bat`**. Au premier lancement, un accueil explique par où commencer.

- Le premier lancement du moteur Plus prépare des moteurs TensorRT pour la carte graphique :
  plusieurs minutes, une seule fois.
- Source vidéo : page **Connexion → Source vidéo**, puis **Webcam du PC** ou **Téléphone**.
- Les autres logiciels (Snapchat, Discord, OBS…) prennent la caméra **« OBS Virtual Camera »**.
- Tout le reste est expliqué dans `LISEZMOI.txt`.

## Téléphone (facultatif)

Android 9 ou plus récent. Installer **`Appli Android\PhoneCam.apk`** sur le téléphone :

- le copier sur le téléphone et l'ouvrir, en autorisant l'installation d'applis inconnues ;
- ou, avec le câble et le débogage USB activé :
  `Programme\tools\platform-tools\adb.exe install "Appli Android\PhoneCam.apk"`.

Puis **Connexion → Connecter par câble**, ou scanner le QR code en Wi-Fi.

## Micro simulé et voix transformée (facultatif)

Installer [VB-Cable A+B](https://vb-audio.com/Cable/) (entrées « CABLE-A » et « CABLE-B ») : les
autres logiciels prennent « CABLE-B Output » comme micro.

## Contenu

| Dossier | Contenu |
| --- | --- |
| `Lancer PhoneCam.bat` | le logiciel |
| `Installer.bat` | l'installation (une fois) |
| `LISEZMOI.txt` | le mode d'emploi complet |
| `Voix\` | les voix du changeur de voix (deux fournies) |
| `Presets\` | les presets fournis (« Par défaut », « Visage Finnyzyy ») |
| `Captures\` | créé à l'installation : photos, enregistrements, captures de diagnostic |
| `Appli Android\` | l'appli du téléphone, déjà compilée |
| `Programme\` | le moteur : code de PhoneCam, Deep-Live-Cam, paquets ; rien à y toucher |

Dans `Programme\` : `PhoneCam\` (code Python, page web, et `android\` : code de l'appli, à compiler
avec Android Studio ou JDK 17 + SDK : `gradle assembleDebug`), `app\` (Deep-Live-Cam : détection,
inswapper, GFPGAN / GPEN ; `Lancer Deep-Live-Cam.bat` ouvre son interface d'origine, facultatif),
`wheels\` (insightface déjà compilé, évite d'installer Visual Studio), `requirements.txt` (versions
exactes des paquets Python).

Créés par l'installateur et jamais envoyés en ligne : `Programme\venv\` (Python), `Programme\data\`
(modèles, réglages, caches), `Programme\tools\` (adb). Désinstaller : supprimer le dossier. Python,
OBS et ffmpeg se désinstallent depuis les Paramètres de Windows.

## Licences

Deep-Live-Cam est sous licence AGPL-3.0 (`Programme\app\LICENSE`). Les modèles viennent de FaceFusion
(github.com/facefusion/facefusion-assets) et de Deep-Live-Cam, chacun sous sa propre licence ;
plusieurs (HyperSwap, inswapper…) sont réservés à un usage non commercial. Voix : RVC
(Retrieval-based-Voice-Conversion-WebUI, MIT), RMVPE et HuBERT / ContentVec (téléchargés depuis
huggingface.co/lj1995/VoiceConversionWebUI), Silero VAD (MIT), GTCRN (MIT). Ne s'en servir qu'avec
l'accord des personnes dont on utilise le visage ou la voix.
