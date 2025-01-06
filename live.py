import cv2
import numpy as np
import joblib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat
import math
from concurrent.futures import ThreadPoolExecutor
import time
import re
from unidecode import unidecode
from deepmultilingualpunctuation import PunctuationModel


###  Loads models and dico

## Letter prediction

# Charger le modèle mediapipe
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Charger le modèle et les encodeurs sauvegardés
scaler_1 = joblib.load("scaler_1.pkl")
scaler_2 = joblib.load("scaler_2.pkl")
best_model_1 = joblib.load("best_model_1.pkl")
best_model_2 = joblib.load("best_model_2.pkl")
best_model_3 = joblib.load("best_model_3.pkl")

## Subtitiles prediction

with open("dictionnaire.txt", 'r', encoding='latin1') as file1:
    with open("lexiqueorgdico.txt", 'r', encoding='latin1') as file2:
        file = list(set(file1).union(set(file2)))
        french_words = [unidecode(word.lower().strip()) for word in file if all(c in "abcdefghijklmnopqrstuvwxyzéèàçù" for c in word.strip().lower())]

MAX_WORD_LENGTH = len(max(french_words, key=len))

model = PunctuationModel()

### Functions

## Letter prediction functions

def get_coords(point):
  return [point.x, point.y, point.z]

def get_angles(base, finger, hand):
    # Convertir les coordonnées des points en un tableau NumPy pour faciliter les calculs
    points = np.array([[base.x, base.y, base.z]] + [[p.x, p.y, p.z] for p in finger])

    # Initialiser une liste pour stocker les angles
    angles = []

    # Boucler sur les triplets de points
    for i in range(len(points) - 2):
        # Points A, B, C
        A, B, C = points[i], points[i + 1], points[i + 2]

        # Calcul des vecteurs BA et BC
        BA = A - B
        BC = C - B

        # Normalisation des vecteurs
        BA = BA / np.linalg.norm(BA)
        BC = BC / np.linalg.norm(BC)

        # Produit scalaire entre BA et BC
        dot_product = np.dot(BA, BC)

        # Calcul de l'angle en radians
        cos_theta = np.clip(dot_product, -1.0, 1.0)  # S'assurer que cos_theta reste dans [-1, 1]

        # Ajouter l'angle en radians à la liste
        angles.append(np.arccos(cos_theta) / np.pi) # Calcul de l'angle en radian / pi (pour que angle soit dans [0, 1])

        if i != len(points) - 3: # Calculer la rotation entre les plans
          # Points A, B, C, D
          D = points[i + 3]

          # Calcul des vecteurs AB, AC, BC, BD
          AB = B - A
          AC = C - A
          BD = D - B
          CD = D - C

          #  Normalisation des vecteurs
          AB = AB / np.linalg.norm(AB)
          AC = AC / np.linalg.norm(AC)
          BD = BD / np.linalg.norm(BD)
          CD = CD / np.linalg.norm(CD)

          # Calcul des vecteurs normaux au plan ABC  et BCD
          u = np.cross(AB, AC)
          v = np.cross(BD, CD)

          # Calcul de l'angle en radians
          cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
          cos_theta = np.clip(cos_theta, -1.0, 1.0)  # S'assurer que cos_theta reste dans [-1, 1]

          # Ajouter l'angle en radians à la liste
          angles.append(np.arccos(cos_theta) / np.pi) # Calcul de l'angle en radian / pi (pour que angle soit dans [0, 1])

    return angles

def signed_distance(point, normal_vector, d, index):
    """
    Calcule la distance signée d'un point par rapport à un plan.
    :param point: Coordonnées du point [x, y, z]
    :param normal_vector: Vecteur normal du plan [a, b, c]
    :param d: Terme constant de l'équation du plan
    :return: Distance signée
    """
    return (np.dot(normal_vector, point) + d) * math.sqrt((index[1].x - index[0].x)**2 + (index[1].y - index[0].y)**2 + (index[1].z - index[0].z)**2) # De la même longueur que la métacarpe de l'index

def projection_point(p1, p2, q):
    """ Calcule la projection du point q sur la droite passant par p1 et p2 """
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(q) - np.array(p2)
    t = np.dot(v1, v2) / np.dot(v1, v1)
    return np.array(p2) + t * v1

def preprocess(detection):
  hand = detection.handedness[0][0].category_name
  base_poignet = detection.hand_landmarks[0][0]
  pouce = detection.hand_landmarks[0][1:5]
  index = detection.hand_landmarks[0][5:9]
  majeur = detection.hand_landmarks[0][9:13]
  annulaire = detection.hand_landmarks[0][13:17]
  auriculaire = detection.hand_landmarks[0][17:21]

  ## Calcul des angles entre les phalages et la rotation
  angles = get_angles(base_poignet, pouce, hand) + get_angles(base_poignet, index, hand) + get_angles(base_poignet, majeur, hand) + get_angles(base_poignet, annulaire, hand) + get_angles(base_poignet, auriculaire, hand)

  ## Ajout de l'orientation de poignet
  hand_without_fingers = np.array([get_coords(base_poignet), get_coords(pouce[0]), get_coords(index[0]), get_coords(majeur[0]), get_coords(annulaire[0]), get_coords(auriculaire[0])])

  # Appliquer la SVD
  U, S, Vt = np.linalg.svd(hand_without_fingers - np.mean(hand_without_fingers, axis=0))

  # Le vecteur normal du plan est donné par la dernière ligne de Vt
  if np.dot(Vt[2], [1, 1, 1]) >= -np.dot(Vt[2], [1, 1, 1]):
    normal_vector = Vt[2]
  else:
    normal_vector = -Vt[2]

  # Si la paume est vers l'avant de l'écran, c'est à dire le point moyen des bout de doigts est dans la même direction que le vecteur normal
  paume_vers_avant = 0
  if signed_distance(np.mean([get_coords(pouce[-1]), get_coords(index[-1]), get_coords(majeur[-1]), get_coords(annulaire[-1]), get_coords(auriculaire[-1])], axis=0), normal_vector, -np.dot(normal_vector, np.mean(hand_without_fingers, axis=0)), index) > 0:
    normal_vector = -normal_vector
    paume_vers_avant = 1

  # Normalisation du vecteur normal
  normal_vector = normal_vector / np.linalg.norm(normal_vector)

  ### L'index et le majeur sont croisés (pour différencier le R du U)

  # Equation du plan moyen
  a, b, c = normal_vector
  d = -np.dot(normal_vector, np.mean(hand_without_fingers, axis=0))

  ## Calcul de la distance signé entre le plan et l'index et le majeur * la distance de la métacarpe de l'index
  majeur_sur_index = signed_distance(get_coords(majeur[-1]), normal_vector, d, index) - signed_distance(get_coords(index[-1]), normal_vector, d, index)

  ## Calcul de la distance entre les points projetés de la dernière phalange de l'index et du majeur sur la droite donnée par les premières phalanges de l'index et du majeur divisé par la distance entre la première distance de l'index et du majeur

  # Projections des derniers points de l'index et du majeur sur la droite
  index_proj = np.array(projection_point(get_coords(index[0]), get_coords(index[1]), get_coords(index[-1])))
  majeur_proj = np.array(projection_point(get_coords(majeur[0]), get_coords(majeur[1]), get_coords(majeur[-1])))

  # Distance entre les projections des derniers points de l'index et du majeur
  distance_proj = np.linalg.norm(index_proj - majeur_proj)

  # Distance entre les premières phalanges de l'index et du majeur
  distance_base = np.linalg.norm(np.array(get_coords(index[0])) - np.array(get_coords(majeur[0])))

  # Calcul final
  ratio = -distance_proj / distance_base

  # Detection de si le pouce est devant l'index (pour le F et le T)
  index_base_proj_ind_aur = projection_point(get_coords(index[0]), get_coords(auriculaire[0]), get_coords(index[0]))
  index_proj_ind_aur = projection_point(get_coords(index[0]), get_coords(auriculaire[0]), get_coords(index[-1]))
  pouce_proj_ind_aur = projection_point(get_coords(index[0]), get_coords(auriculaire[0]), get_coords(pouce[-1]))
  pouce_ind_croise = int(np.linalg.norm(index_base_proj_ind_aur - index_proj_ind_aur) > np.linalg.norm(index_base_proj_ind_aur - pouce_proj_ind_aur)) # 1 si croise, 0 sinon

  ## Calculer l'angle entre l'index et le majeur

  # Convertir les points en numpy arrays

  # Calcul des vecteurs AB et CD
  index_top_bottom = np.array(get_coords(index[-1])) - np.array(get_coords(index[0]))
  majeur_top_bottom = np.array(get_coords(majeur[-1])) - np.array(get_coords(majeur[0]))

  # Produit scalaire des deux vecteurs
  dot_product = np.dot(index_top_bottom, majeur_top_bottom)

  # Cosinus de l'angle
  cos_theta = dot_product / (np.linalg.norm(index_top_bottom) * np.linalg.norm(majeur_top_bottom))

  # Angle en radians
  angle_index_majeur = np.arccos(np.clip(cos_theta, -1.0, 1.0)) / np.pi  # Clip pour éviter les erreurs numériques

  return angles + [paume_vers_avant, majeur_sur_index, ratio, pouce_ind_croise, angle_index_majeur]

def predict(rgb_frame):
    """
    Prédit le label à partir de l'image
    """
    try:
        detection = detector.detect(Image(image_format=ImageFormat.SRGB, data=rgb_frame))
        if len(detection.handedness) == 0:
            return None

        if detection.handedness[0][0].category_name == "Left":
            for j in range(0, len(detection.hand_landmarks[0])):
                detection.hand_landmarks[0][j].x = 1 - detection.hand_landmarks[0][j].x

        preprocessed = preprocess(detection)

        data_model_1 = scaler_1.transform([preprocessed])
        y_pred_1 = best_model_1.predict_proba(data_model_1)
        entropy_1 = -np.sum(np.clip(y_pred_1[0], 1e-10, 1) * np.log(np.clip(y_pred_1[0], 1e-10, 1))) / math.log(len(y_pred_1[0]))
        if entropy_1 > 0.8:
            return None
        
        data_model_2 = scaler_2.transform([np.array([[lm.x, lm.y, lm.z] for lm in detection.hand_landmarks[0]]).flatten()])
        y_pred_2 = best_model_2.predict_proba(data_model_2)
        entropy_2 = -np.sum(np.clip(y_pred_2[0], 1e-10, 1) * np.log(np.clip(y_pred_2[0], 1e-10, 1))) / math.log(len(y_pred_2[0]))
        if  entropy_2 > 0.9:
            return None

        data_model_3 = np.concatenate((y_pred_1, y_pred_2), axis=1)
        y_pred_3 = best_model_3.predict(data_model_3)[0]

        return chr(65 + y_pred_3)
    except Exception as e:
        print(e)
        return None
    
###  Subtitiles prediction functions
def maxMatch(string):
    tokens = []
    not_in = ""
    i = 0
    while i < len(string):
        maxWord = ""
        for j in range(i, len(string)):
            tempWord = string[i:j+1]
            if tempWord in french_words and len(tempWord) > len(maxWord):
                maxWord = tempWord
        if len(maxWord) == 0:
            not_in = string[i:j+1]
            break
        i = i+len(maxWord)
        tokens.append(maxWord)
    return tokens, not_in

def completeMaxMatch(string):
    tokens, not_in = maxMatch(string)
    not_in_array = [0] * len(tokens)
    while len(not_in) > 0:
        tokens.append(not_in[0])
        not_in_array.append(1)
        not_in  = not_in[1:]
        if len(not_in) > 0:
            tokens_, not_in = maxMatch(not_in)
            tokens.extend(tokens_)
            not_in_array.extend([0] * len(tokens_))
    return  tokens, not_in_array

def reverseMaxMatch(string):
    tokens = []
    not_in = ""
    i = len(string)
    while i > 0:
        maxWord = ""
        for j in range(i - 1, -1, -1):
            tempWord = string[j:i]
            if tempWord in french_words and len(tempWord) > len(maxWord):
                maxWord = tempWord
        if len(maxWord) == 0:
            not_in = string[j:i]
            break
        i = i - len(maxWord)
        tokens.append(maxWord)
    return tokens[::-1], not_in

def completeReverseMaxMatch(string):
    tokens, not_in = reverseMaxMatch(string)
    not_in_array = [0] * len(tokens)
    while len(not_in) > 0:
        tokens.insert(0, not_in[-1])
        not_in_array.insert(0, 1)
        not_in  = not_in[:-1]
        if len(not_in) > 0:
            tokens_, not_in = reverseMaxMatch(not_in)
            tokens = tokens_ + tokens 
            not_in_array = [0] * len(tokens_) + not_in_array
    return tokens, not_in_array

def get_splits_idx(all_tokens_l):
    l_splits_idx = []
    sum_ = 0
    for token in all_tokens_l:
        sum_ += len(token)
        l_splits_idx.append(sum_)
    return l_splits_idx

def compare_tokens_list(all_tokens_l1, not_in_array_l1, all_tokens_l2, not_in_array_l2):
    if sum(not_in_array_l1) <  sum(not_in_array_l2):
        return all_tokens_l1
    elif sum(not_in_array_l1) > sum(not_in_array_l2):
        return all_tokens_l2
    else:
        sum_square_len_l1 = sum([len(word) ** 2 for word in all_tokens_l1])
        sum_square_len_l2 = sum([len(word) ** 2 for word in all_tokens_l2])
        return all_tokens_l1 if sum_square_len_l1 > sum_square_len_l2 else all_tokens_l2

def mix_algo(string):
    tokens_max, not_in_array_max = completeMaxMatch(string)
    tokens_reverse_max, not_in_array_reverse_max = completeReverseMaxMatch(string)

    max_splits_idx = get_splits_idx(tokens_max)
    reverse_max_splits_idx = get_splits_idx(tokens_reverse_max)
    
    finals_words = []
    i_act = 0
    j_act = 0
    for i in range(len(max_splits_idx)):
        for j in range(j_act, len(reverse_max_splits_idx)):
            if max_splits_idx[i] == reverse_max_splits_idx[j]:
                finals_words.extend(compare_tokens_list(tokens_max[i_act : i + 1], not_in_array_max[i_act : i + 1], tokens_reverse_max[j_act : j + 1], not_in_array_reverse_max[j_act : j + 1]))
                i_act = i + 1
                j_act = j + 1
                break
    return finals_words

def algo_optimise(string):
    confirmed_ = []
    next_ = []
    next_raw  = []
    i = 0
    while True :
        next_ = mix_algo(string[i:i+MAX_WORD_LENGTH])
        if i + MAX_WORD_LENGTH >= len(string):
            next_raw = string[i:i+MAX_WORD_LENGTH]
            break
        else:
            confirmed_.append(next_[0])
            i += len(next_[0])
        
    return confirmed_, next_, next_raw

def capitalize_sentences(text):
    # Utilisation de re.sub pour identifier chaque phrase et mettre la première lettre en majuscule
    return re.sub(r'(^|(?<=[.!?…])\s+)([a-z])', lambda match: match.group(1) + match.group(2).upper(), text)

def tokens_to_text(tokens):
    return capitalize_sentences(model.restore_punctuation(" ".join(tokens)))

## Main function of the code
def main():
    cap = cv2.VideoCapture(0)
    current_label = None
    quitted = False
    next_raw = ""
    next_ = []
    confirmed = []
    prediction_subtitles = None
    prediction_tokens_to_text = None
    subtitles =  ""

    if not cap.isOpened():
        print("Erreur : Impossible d'accéder à la webcam.")
        return

    try:
        prediction_start = time.time()
        last_prediction = ""
        while True:
            ret, frame = cap.read()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not ret:
                print("Erreur : Impossible de lire la vidéo.")
                break

            with ThreadPoolExecutor() as executor:
                prediction = executor.submit(predict, rgb_frame)

                while not prediction.done():
                    if prediction_subtitles is not None and prediction_subtitles.done():
                        confirmed_, next_, next_raw = prediction_subtitles.result()
                        confirmed.extend(confirmed_)
                        prediction_subtitles = None
                        prediction_tokens_to_text = executor.submit(tokens_to_text, confirmed + next_)

                    if prediction_tokens_to_text is not None and prediction_tokens_to_text.done():
                        subtitles = prediction_tokens_to_text.result()
                        prediction_tokens_to_text = None

                    if current_label is not None:
                        cv2.putText(frame, f"Prediction: {current_label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                        delay = math.ceil(3 + prediction_start - time.time())
                        delay_str = str(delay)
                        if delay > 0:
                            (text_width, text_height), _ = cv2.getTextSize(delay_str, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                            text_x = (frame.shape[1] - text_width) // 2
                            text_y = (frame.shape[0] - text_height) // 2
                            cv2.putText(frame, delay_str, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        else:
                            next_raw += last_prediction.lower()
                            subtitles += last_prediction.lower()
                            prediction_subtitles = executor.submit(algo_optimise, next_raw)
                            prediction_start = time.time()

                    if len(subtitles) > 0:
                        i = 0
                        (text_width, text_height), _ = cv2.getTextSize(subtitles, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        while text_width > 0.80 * frame.shape[1]:
                            i += 1
                            (text_width, text_height), _ = cv2.getTextSize(" ".join(subtitles.split(" ")[i:]), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        text_x = (frame.shape[1] - text_width) // 2
                        text_y = int(frame.shape[0] * 0.90 -  text_height // 2)
                        cv2.putText(frame, " ".join(subtitles.split(" ")[i:]), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    cv2.imshow('Real-time Hand Detection and Prediction', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        quitted = True
                        break
                    ret, frame = cap.read()
                    if not ret:
                        print("Erreur : Impossible de lire la vidéo.")
                        break

            if quitted:
                break
            
            current_label = prediction.result()

            if current_label != last_prediction:
                last_prediction  = current_label
                prediction_start  =  time.time()

    finally:
        with open("subtitles.txt", "w") as file:
            file.write(subtitles)
        print("Transcript saved in subtitles.txt")

        print("Libération des ressources...")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()