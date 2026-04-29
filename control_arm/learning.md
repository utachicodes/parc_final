# learning.md — Comprendre les technologies du projet

Projet : SO-101 Robot Controller (`app.py` + `templates/index.html`)

---

## Domaine : Robotique embarquée + Vision par ordinateur + Interface web temps réel

Ce projet se situe à l'intersection de **trois disciplines** :

| Discipline | Ce qu'on fait dans ce projet |
|---|---|
| **Robotique** | Commander un bras 6-DOF via liaison série (protocole propriétaire) |
| **Vision par ordinateur** | Détecter un visage/main en temps réel depuis une webcam |
| **Développement web embarqué** | Servir une UI de contrôle depuis le robot lui-même (Flask sur localhost) |

C'est ce qu'on appelle de la **robotique interactable** : le robot réagit à son environnement visuel ET peut être commandé manuellement via une interface web, tout en tournant sur une seule machine Python.

---

## Les bibliothèques utilisées et pourquoi

---

### 1. Flask

**Ce que c'est** : micro-framework web Python. Transforme un script Python en serveur HTTP.

**Pourquoi ici** :
- Sert l'interface de contrôle (`index.html`) depuis `localhost:5000`
- Expose des endpoints REST (`/api/connect`, `/api/servo/move`, etc.) appelés en JS depuis le navigateur
- Permet de streamer la vidéo webcam (`/video_feed`) et les logs (`/stream_logs`) en HTTP

**Pourquoi Flask et pas autre chose** :
- Zéro configuration, démarrage en 5 lignes — parfait pour un projet de démonstration
- Pas besoin d'un vrai serveur de production (nginx, gunicorn) pour un usage local
- Jinja2 (templating HTML) intégré, ce qui génère le tableau des servos dynamiquement depuis `SERVO_IDS`

**Alternatives** :
| Alternative | Différence clé |
|---|---|
| **FastAPI** | Plus moderne, support async natif, meilleur pour les APIs grandes. Plus complexe à démarrer. |
| **Django** | Full-stack avec ORM, admin, auth. Overkill ici — trop lourd pour contrôler un robot. |
| **aiohttp** | Entièrement asynchrone. Utile si on voulait du WebSocket natif, mais plus complexe. |
| **ROS 2 (rosbridge)** | Standard industrie robotique. Beaucoup plus puissant mais énorme courbe d'apprentissage. |

---

### 2. OpenCV (`cv2`)

**Ce que c'est** : bibliothèque C++ avec bindings Python pour le traitement d'images et la vision par ordinateur.

**Pourquoi ici** :
- Capture les frames de la webcam (`cv2.VideoCapture`)
- Encode chaque frame en JPEG pour le stream MJPEG (`cv2.imencode`)
- Convertit les espaces de couleur (`cv2.cvtColor` BGR→RGB pour MediaPipe)
- Fallback de détection visage via Haar Cascades (`cv2.CascadeClassifier`) si MediaPipe échoue

**Pourquoi OpenCV et pas autre chose** :
- Référence absolue en vision par ordinateur depuis 20 ans
- Support matériel excellent (webcams USB, Raspberry Pi Camera, etc.)
- Intégration native avec NumPy : une frame = un `np.ndarray`

**Alternatives** :
| Alternative | Différence clé |
|---|---|
| **Pillow (PIL)** | Manipulation d'images statiques uniquement. Pas de capture vidéo. |
| **imageio** | Lecture/écriture fichiers. Pas de traitement temps réel. |
| **PyCamera2** | Spécifique Raspberry Pi Camera. Non portable. |
| **ffmpeg-python** | Encodage/streaming vidéo puissant, mais pas de traitement frame par frame facile. |

---

### 3. NumPy (`numpy`)

**Ce que c'est** : bibliothèque de calcul numérique — tableaux multidimensionnels performants et opérations mathématiques vectorisées.

**Pourquoi ici** :
- `np.clip(out, lo, hi)` dans le PID controller pour borner la sortie en une ligne
- Les frames vidéo OpenCV sont des `np.ndarray` — NumPy est une dépendance implicite d'OpenCV

**Pourquoi NumPy et pas `math` standard** :
- `np.clip` est plus lisible et fonctionne aussi bien sur un scalaire que sur un tableau
- Dans un projet de vision, NumPy est déjà là de toute façon (dépendance d'OpenCV et MediaPipe)

**Alternatives** :
- `min(hi, max(lo, out))` en Python pur — fonctionne mais moins expressif
- `torch.clamp` si on était dans un pipeline PyTorch

---

### 4. MediaPipe

**Ce que c'est** : bibliothèque Google de ML embarqué. Fournit des modèles de détection (visage, main, pose, etc.) optimisés pour tourner en temps réel sur CPU.

**Pourquoi ici** :
- Détecte les visages (`blaze_face_short_range.tflite`) pour le tracking automatique
- Détecte les mains (`hand_landmarker.task`) pour le contrôle gestuel
- Les modèles `.tflite` sont téléchargés une seule fois et tournent localement — pas d'API cloud

**Pourquoi MediaPipe et pas autre chose** :
- Temps réel sur CPU sans GPU : les modèles TFLite sont ultra-optimisés (~5-15ms par frame)
- Pas besoin de GPU — important sur un PC standard ou futur portage Raspberry Pi
- API Tasks moderne (vs ancienne API legacy) : `FaceDetector.create_from_options()`

**Alternatives** :
| Alternative | Différence clé |
|---|---|
| **OpenCV Haar Cascades** | Déjà utilisé comme fallback. Très rapide mais faux positifs fréquents, pas de landmarks. |
| **face_recognition** | Basé sur dlib. Très précis pour identification, mais lent (~200ms). Trop lent pour 30fps. |
| **YOLO (ultralytics)** | Détection générale d'objets très puissante. Nécessite GPU pour temps réel. |
| **InsightFace** | Précision maximale. Lourd, nécessite GPU. |
| **Dlib** | Landmarks faciaux 68 points. Précis mais 3-5x plus lent que MediaPipe. |

---

### 5. scservo_sdk

**Ce que c'est** : SDK Python officiel Feetech pour contrôler les servomoteurs SMS/STS via port série (RS-485 ou TTL demi-duplex).

**Pourquoi ici** :
- C'est le **seul moyen** de parler aux servos Feetech STS3215 du SO-101
- Implémente le protocole paquet binaire Feetech (header, ID, longueur, instruction, checksum)
- Fonctions clés utilisées :
  - `WritePosEx(id, pos, speed, acc)` — commande de position
  - `WriteSpec(id, speed, acc)` — commande de vitesse (mode roue)
  - `ReadPos(id)` — lecture position courante
  - `read2ByteTxRx(id, addr)` — lecture registre EEPROM 2 octets
  - `read1ByteTxRx(id, addr)` — lecture registre EEPROM 1 octet

**Pourquoi ce SDK et pas autre chose** :
- C'est le SDK fourni par le fabricant — pas d'alternative officielle
- Les servos Feetech ont un protocole propriétaire (inspiré de Dynamixel mais incompatible)

**Alternatives si on changeait de servos** :
| Servo | SDK |
|---|---|
| **Dynamixel** (Robotis) | `dynamixel_sdk` — protocole similaire, meilleure documentation |
| **Servo RC standard** | `RPi.GPIO` + PWM, ou carte PCA9685 via `adafruit-pca9685` |
| **Moteur DC** | `pyserial` + Arduino intermédiaire |
| **ROS 2** | `ros2_control` — abstrait tous les types d'actionneurs |

---

### 6. `threading` + `queue` (stdlib Python)

**Ce que c'est** : modules standard Python pour le parallélisme par threads et la communication inter-thread sans data race.

**Pourquoi ici** :
- `TrackerThread(threading.Thread)` : tourne à 30fps indépendamment du serveur Flask
- `threading.Lock()` sur le port série : un seul thread parle aux servos à la fois (voir Bug 4)
- `threading.Event()` pour l'arrêt de calibration : signal propre entre threads sans `while True`
- `queue.Queue` pour les logs SSE : le thread de log écrit, les clients HTTP lisent, sans corruption

**Pourquoi threads et pas async/await** :
- Flask standard est synchrone — mélanger `asyncio` et Flask WSGI est compliqué
- Les opérations bloquantes (port série, OpenCV) fonctionnent mieux avec de vrais threads qu'avec de l'async I/O
- `threading.Lock` est simple et suffisant pour ce niveau de concurrence

**Alternatives** :
| Alternative | Différence clé |
|---|---|
| **multiprocessing** | Vrais processus séparés, contourne le GIL Python. Overkill ici. |
| **asyncio** | Meilleur pour I/O réseau massivement concurrent. Complexe avec code bloquant. |
| **concurrent.futures** | Abstraction de threads/processus. Plus haut niveau mais moins de contrôle. |
| **Celery** | Queue de tâches distribuée. Très overkill pour un processus local. |

---

### 7. SSE — Server-Sent Events

**Ce que c'est** : technologie HTTP standard pour qu'un serveur envoie des événements en continu vers un navigateur, sans que le client redemande à chaque fois.

**Pourquoi ici** :
- Les logs (`/stream_logs`) et les stats (`/stream_stats`) sont poussés vers l'UI en temps réel
- Le navigateur ouvre une connexion `EventSource` et reçoit les messages au fur et à mesure
- Utilisé pour afficher : logs du robot, état connecté/déconnecté, statut calibration, FPS

**Pourquoi SSE et pas WebSocket** :
- SSE est **unidirectionnel** (serveur → client) : parfait pour des logs et stats
- Plus simple à implémenter avec Flask (`Response` + `stream_with_context` + `text/event-stream`)
- WebSocket nécessite une librairie supplémentaire (`flask-socketio`) et est bidirectionnel (utile seulement si le client doit aussi envoyer des flux continus)

**Alternatives** :
| Alternative | Différence clé |
|---|---|
| **WebSocket** | Bidirectionnel. Nécessaire si le client envoie du streaming (audio, vidéo, joystick). |
| **Polling AJAX** | Le client redemande toutes les X secondes. Simple mais inefficace et avec délai. |
| **Long-polling** | Le client garde une connexion ouverte en attente. Hacky, déprécié. |
| **MQTT** | Protocole pub/sub léger, standard IoT. Excellent mais nécessite un broker externe. |

---

### 8. Jinja2 (via Flask)

**Ce que c'est** : moteur de templates HTML intégré à Flask. Permet d'injecter des variables Python dans le HTML.

**Pourquoi ici** :
- `{% for name in servo_ids.keys() %}` génère automatiquement une ligne par servo dans le tableau
- Si on ajoute un servo dans `servo_ids.json`, le tableau UI se met à jour sans toucher au HTML

**Alternatives** :
- HTML statique + JS qui appelle `/api/servo_ids` au chargement — plus de séparation front/back
- React/Vue.js — frameworks JS modernes, bien plus puissants pour des UIs complexes, mais overkill ici

---

## Protocole série des servos Feetech — concepts clés

Les servos communiquent via un bus RS-485 ou TTL demi-duplex à 1 Mbaud. Chaque servo a un **ID unique** (1–253). Le protocole est paquet binaire :

```
[0xFF][0xFF][ID][LEN][INSTRUCTION][PARAMS...][CHECKSUM]
```

**Registres importants (EEPROM)** :
| Adresse | Nom | Taille | Description |
|---|---|---|---|
| 9 | MIN_ANGLE_LIMIT | 2 octets | Position minimum autorisée (0–4095) |
| 11 | MAX_ANGLE_LIMIT | 2 octets | Position maximum autorisée (0–4095) |
| 33 | MODE | 1 octet | 0=position, 1=roue (vitesse), 2=pas à pas |
| 40 | TORQUE_ENABLE | 1 octet | 0=libre, 1=actif |

**Conversion degrés ↔ unités brutes** :
```
position_brute = 2048 + degrés × (4095 / 360)
degrés = (position_brute - 2048) / (4095 / 360)
```
- `2048` = position centrale (0°)
- `0` = -180°, `4095` = +180°

---

## PID Controller — principe

Un **PID** (Proportionnel-Intégral-Dérivé) est un algorithme de contrôle en boucle fermée.

Dans ce projet, deux PIDs contrôlent le pan (rotation horizontale) et le tilt (rotation verticale) pour que le robot suive un visage :

```
erreur  = position_cible - position_actuelle   (ex: visage à gauche → erreur négative)
sortie  = Kp × erreur + Ki × ∫erreur + Kd × (Δerreur / Δt)
```

| Terme | Rôle | Trop fort = |
|---|---|---|
| **Kp** (proportionnel) | Réagit à l'erreur courante | Oscillations |
| **Ki** (intégral) | Corrige l'erreur persistante | Dépassement lent |
| **Kd** (dérivé) | Amortit les changements rapides | Sensible au bruit |

La sortie est clampée entre `-pan_range` et `+pan_range` (en degrés) pour ne pas dépasser les limites physiques du servo.

---

## Récap des domaines de connaissance impliqués

| Domaine | Technologies | Niveau requis |
|---|---|---|
| Robotique / actionneurs | scservo_sdk, protocole série RS-485, registres EEPROM | Intermédiaire |
| Vision par ordinateur | OpenCV, MediaPipe, espaces couleur BGR/RGB | Débutant–intermédiaire |
| Contrôle automatique | PID controller, boucle fermée, tuning Kp/Ki/Kd | Débutant |
| Programmation concurrente | threading, Lock, Event, Queue | Intermédiaire |
| Développement web back-end | Flask, REST API, SSE, Jinja2 | Débutant |
| Développement web front-end | HTML/CSS/JS vanilla, fetch API, EventSource | Débutant |
| Systèmes embarqués | Port série, baudrate, demi-duplex, timing | Débutant–intermédiaire |
