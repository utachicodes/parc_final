# Maux de tête — Journal des bugs & résolutions

Projet : SO-101 Robot Controller (`app.py` + `templates/index.html`)

---

## Bug 1 — `api_servo_move` retournait toujours `ok: true`

**Symptôme** : cliquer sur ↑/↓/● dans le tableau Servo Limits ne donnait aucun retour d'erreur, même quand la commande échouait silencieusement.

**Cause** : `write_raw_position` retournait `False` en cas d'échec mais `api_servo_move` ignorait ce retour et renvoyait toujours `{'ok': True}`. Pareil pour `write_raw_position` qui ne loggait rien si le robot n'était pas connecté.

**Résolution** :
- `write_raw_position` logge maintenant `"write_raw IDx: robot non connecté"` si `self.connected` est False.
- `api_servo_move` vérifie le retour et renvoie `{'ok': False, msg: '...'}` en cas d'échec → le toast rouge s'affiche dans l'UI.

---

## Bug 2 — DummyRobot remplaçait le vrai Robot dans `state`

**Symptôme** : après avoir démarré le tracking sans connexion active, puis s'être reconnecté, les commandes manuelles ne faisaient rien. Aucun message d'erreur.

**Cause** : dans `api_tracking_start`, si `state['robot']` n'était pas connecté, le code faisait `state['robot'] = DummyRobot()`. `DummyRobot.write_raw_position` fait `pass` silencieusement. Même après une reconnexion, `state['robot']` pointait encore vers le DummyRobot si le tracking n'avait pas été relancé.

**Résolution** : le DummyRobot est maintenant **local au tracker** uniquement. `state['robot']` n'est jamais écrasé par un DummyRobot. La reconnexion via `api_connect` remet toujours un vrai `Robot` dans `state['robot']`.

---

## Bug 3 — Le tracker thread overridait les servos fixes à chaque frame

**Symptôme** : `wrist_roll` commandé à 90° ne bougeait jamais (ou pendant 33ms puis revenait à 0). Résultat : "ne fait rien".

**Cause** : le `TrackerThread` tourne à 30fps. À chaque frame avec détection, il envoie **tous** les servos dont les "joints fixes" (`wrist_roll`, `elbow_flex`, `wrist_flex`, `gripper`) à la valeur de `self.s` (= 0 par défaut). Toute commande manuelle était écrasée dans les 33ms suivantes.

**Résolution en deux parties** :

1. `api_tracking_start` passe maintenant `state['settings']` **par référence** (pas une copie) au `TrackerThread`. Ainsi `self.s` dans le tracker IS `state['settings']` — toute modification UI est prise en compte immédiatement sans redémarrer le tracking.

2. `api_servo_move` met à jour `state['settings'][name]` pour les joints fixes (`elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`) après une commande réussie. Le tracker lit la nouvelle valeur dès la frame suivante au lieu de remettre à 0.

3. La fonction JS `moveToLimit` synchronise aussi le slider "Joints fixes" dans l'UI et appelle `pushSettings()` pour confirmer côté serveur.

---

## Bug 4 — Race condition sur le port série (comportement aléatoire)

**Symptôme** : parfois la commande passait, parfois non. Aléatoire. Impossible à reproduire de façon fiable.

**Cause** : le port série (`self.servo.WritePosEx`, `ReadPos`) était accédé depuis **plusieurs threads simultanément** sans aucun verrou :
- Le `TrackerThread` envoie ~180 paquets/sec (6 servos × 30fps)
- Les handlers Flask (`api_servo_move`, `api_servo_read`, `api_servo_hw_limits`) écrivent/lisent aussi

Quand deux threads écrivaient en même temps, les paquets série se corrompaient → le servo recevait des données invalides et ignorait la commande.

**Résolution** : ajout d'un `threading.Lock()` (`self._lock`) dans la classe `Robot`. Tous les accès série sont maintenant protégés :
```python
with self._lock:
    self.servo.WritePosEx(...)   # send_positions, write_raw_position
    self.servo.ReadPos(...)      # read_raw_positions, api_servo_read
    robot.servo.read2ByteTxRx(...)  # api_servo_hw_limits
```
Un seul thread à la fois peut accéder au port. Les autres attendent. Plus de comportement intermittent.

---

## Bug 5 — Calibration : wrist_roll invisible + mauvaise architecture

**Symptôme** : le servo 5 (`wrist_roll`) n'apparaissait pas dans les données de calibration. La calibration auto balayait ±500 unités brutes sans tenir compte des limites définies dans le tableau UI.

**Causes** :
- `showCalData` utilisait `data.center_pos` issu uniquement d'un auto-sweep jamais lancé
- L'auto-calibration ignorait le tableau "Servo Limits" de l'UI et calculait ses propres bornes (±500 units depuis position actuelle ≈ ±44°)
- Aucun moyen de sauvegarder/charger les limites du tableau

**Résolution** :
- `auto_calibrate` accepte maintenant `limits_deg` (les valeurs du tableau UI en degrés) et les utilise comme bornes de sweep
- `saveCalibration` lit le tableau complet (tous servos dont `wrist_roll`) et écrit `calibration.json` avec min/ctr/max en degrés ET en unités brutes
- `loadCalibration` recharge le fichier et remplit automatiquement le tableau Servo Limits dans l'UI
- Boutons ajoutés : **💾 Save Limits**, **📂 Load Cal**, **⏹ Stop** (interrompt le sweep entre deux servos), **↺ Reset** (repositionne tous les servos aux positions center sauvegardées)

---

## Bug 6 — Limites hardware inconnues du servo wrist_roll

**Symptôme** : même avec les bugs logiciels corrigés, certaines positions (ex. 90°) pouvaient être rejetées silencieusement par le firmware du servo sans message d'erreur.

**Cause** : les servos Feetech SMS/STS ont des registres EEPROM `MIN_ANGLE_LIMIT` (adresse 9) et `MAX_ANGLE_LIMIT` (adresse 11) qui définissent la plage physique autorisée. Si on envoie une position hors de cette plage, le servo ignore la commande sans retour d'erreur.

**Résolution** : ajout d'un endpoint `/api/servo/hw_limits` et d'un bouton **HW** (jaune) par servo dans le tableau. En cliquant **HW** :
- Lit les registres EEPROM 9 et 11 via `read2ByteTxRx`
- Affiche la vraie plage dans le log et dans un toast
- Met à jour automatiquement les champs min/max du tableau avec les vraies valeurs hardware

---

## Bug 7 — Settings figés dans le tracker thread (wrist_roll slider sans effet)

**Symptôme** : changer le slider `wrist_roll` dans le panneau "Joints fixes" pendant un tracking actif n'avait aucun effet sur le servo.

**Cause** : `TrackerThread.__init__` copiait `settings` en `self.s = settings` depuis le dict `s = {**state['settings'], **data}` qui était une **copie**, pas une référence. Toute modification ultérieure de `state['settings']` n'était pas visible dans `self.s`.

**Résolution** : voir Bug 3 — `state['settings']` est passé directement par référence. `pushSettings()` met à jour `state['settings']` en place → tracker voit la valeur immédiatement.

---

## Bug 8 — wrist_roll en mode ROUE (wheel mode) → `WritePosEx` ignoré

**Symptôme** : toutes les commandes de position envoyées à `wrist_roll` (ID5) étaient ignorées. Le log confirmait que les bonnes valeurs en unités brutes étaient envoyées (ex. 3071 pour 90°), les limites EEPROM étaient `0–4095` (plage complète), aucune erreur HTTP — mais le servo ne bougeait pas. En revanche, pendant l'auto-calibration (sweep continu), il bougeait.

**Cause** : le servo était configuré en **mode roue** (`MODE register 33 = 1`). Dans ce mode, le firmware Feetech **ignore complètement** `WritePosEx` — la commande de position est silencieusement rejetée. Seule la commande de vitesse (`WriteSpec`) est prise en compte. C'est cohérent avec le rôle du servo : faire tourner le gripper en rotation continue. L'auto-calibration fonctionnait car le sweep interne utilise un mécanisme différent.

**Résolution en deux parties** :

1. **Diagnostic** : ajout de la lecture du registre MODE (adresse 33) dans `/api/servo/hw_limits` via `read1ByteTxRx`. Le bouton **HW** affiche maintenant `MODE=1 (wheel)` dans le log et dans le toast, et révèle automatiquement les contrôles de vitesse.

2. **Contrôle vitesse** : nouvel endpoint `/api/servo/wheel_speed` qui appelle `robot.servo.WriteSpec(sid, speed, acc)` avec `speed` entre -1023 et +1023 (négatif = CCW, positif = CW, 0 = stop). Une barre de contrôle `⟳ wheel` apparaît sous la ligne du servo dans le tableau Servo Limits dès que MODE=1 est détecté :

   | Bouton | Valeur | Direction  |
   |--------|--------|------------|
   | `<<`   | -600   | CCW rapide |
   | `<`    | -200   | CCW lent   |
   | `stop` |    0   | Arret      |
   | `>`    | +200   | CW lent    |
   | `>>`   | +600   | CW rapide  |

---

## Récap technique

| Bug | Fichier | Mécanisme |
| --- | --- | --- |
| Retour d'erreur silencieux | `app.py` | Vérification du retour de `write_raw_position` |
| DummyRobot écrase state | `app.py` | DummyRobot local au tracker uniquement |
| Override 30fps des joints | `app.py` + `index.html` | `state['settings']` par référence + update sur servo/move |
| Race condition série | `app.py` | `threading.Lock()` sur tous les accès `self.servo.*` |
| Calibration sans limites UI | `app.py` + `index.html` | `limits_deg` passé à `auto_calibrate` + save/load complet |
| Limites EEPROM inconnues | `app.py` + `index.html` | Endpoint `hw_limits` + bouton HW |
| Settings copiés au lieu de référencés | `app.py` | Passage de `state['settings']` par référence |
| wrist_roll MODE ROUE — WritePosEx ignoré | `app.py` + `index.html` | Lecture MODE reg 33 + endpoint `wheel_speed` + UI vitesse |
