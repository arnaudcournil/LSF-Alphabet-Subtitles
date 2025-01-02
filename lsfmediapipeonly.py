import cv2
import numpy as np
import mediapipe as mp
import joblib

class SignPredictor:
    def __init__(self):
        # Modèles entrainés et scalers
        self.scaler_1 = joblib.load("scaler_1.pkl")
        self.scaler_2 = joblib.load("scaler_2.pkl")
        self.model_1 = joblib.load("best_model_1.pkl")
        self.model_2 = joblib.load("best_model_2.pkl")
        self.model_3 = joblib.load("best_model_3.pkl")
        
        # Mediapipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("Models loaded successfully")

    def preprocess(self, detection):
        base_poignet = detection.hand_landmarks[0][0]
        pouce = detection.hand_landmarks[0][1:5]
        index = detection.hand_landmarks[0][5:9]
        majeur = detection.hand_landmarks[0][9:13]
        annulaire = detection.hand_landmarks[0][13:17]
        auriculaire = detection.hand_landmarks[0][17:21]

        def get_coords(point):
            return [point.x, point.y, point.z]

        def get_angles(base, finger, hand):
            points = np.array([[base.x, base.y, base.z]] + [[p.x, p.y, p.z] for p in finger])
            angles = []

            for i in range(len(points) - 2):
                A, B, C = points[i], points[i + 1], points[i + 2]
                BA = A - B
                BC = C - B
                BA = BA / np.linalg.norm(BA)
                BC = BC / np.linalg.norm(BC)
                dot_product = np.dot(BA, BC)
                cos_theta = np.clip(dot_product, -1.0, 1.0)
                angles.append(np.arccos(cos_theta) / np.pi)

                if i != len(points) - 3:
                    D = points[i + 3]
                    AB = B - A
                    AC = C - A
                    BD = D - B
                    CD = D - C
                    AB = AB / np.linalg.norm(AB)
                    AC = AC / np.linalg.norm(AC)
                    BD = BD / np.linalg.norm(BD)
                    CD = CD / np.linalg.norm(CD)
                    u = np.cross(AB, AC)
                    v = np.cross(BD, CD)
                    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    angles.append(np.arccos(cos_theta) / np.pi)

            return angles

        def signed_distance(point, normal_vector, d, index):
            return (np.dot(normal_vector, point) + d) * np.sqrt((index[1].x - index[0].x)**2 + (index[1].y - index[0].y)**2 + (index[1].z - index[0].z)**2)

        def projection_point(p1, p2, q):
            v1 = np.array(p1) - np.array(p2)
            v2 = np.array(q) - np.array(p2)
            t = np.dot(v1, v2) / np.dot(v1, v1)
            return np.array(p2) + t * v1

        hand = detection.handedness[0][0].category_name
        
        # Calcul des angles pour les doigts
        angles = get_angles(base_poignet, pouce, hand) + \
                get_angles(base_poignet, index, hand) + \
                get_angles(base_poignet, majeur, hand) + \
                get_angles(base_poignet, annulaire, hand) + \
                get_angles(base_poignet, auriculaire, hand)

        # Calcul main sans les doigts
        hand_without_fingers = np.array([
            get_coords(base_poignet),
            get_coords(pouce[0]),
            get_coords(index[0]),
            get_coords(majeur[0]),
            get_coords(annulaire[0]),
            get_coords(auriculaire[0])
        ])

        # SVD for normal vector
        U, S, Vt = np.linalg.svd(hand_without_fingers - np.mean(hand_without_fingers, axis=0))
        normal_vector = Vt[2] if np.dot(Vt[2], [1, 1, 1]) >= -np.dot(Vt[2], [1, 1, 1]) else -Vt[2]

        # Calcul direction de la paume
        paume_vers_avant = 0
        mean_fingertips = np.mean([
            get_coords(pouce[-1]),
            get_coords(index[-1]),
            get_coords(majeur[-1]),
            get_coords(annulaire[-1]),
            get_coords(auriculaire[-1])
        ], axis=0)

        if signed_distance(mean_fingertips, normal_vector,
                         -np.dot(normal_vector, np.mean(hand_without_fingers, axis=0)), index) > 0:
            normal_vector = -normal_vector
            paume_vers_avant = 1

        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        a, b, c = normal_vector
        d = -np.dot(normal_vector, np.mean(hand_without_fingers, axis=0))

        # Calcul additionel
        majeur_sur_index = signed_distance(get_coords(majeur[-1]), normal_vector, d, index) - \
                          signed_distance(get_coords(index[-1]), normal_vector, d, index)

        # Calcul ratio entre distance des doigts
        index_proj = projection_point(get_coords(index[0]), get_coords(index[1]), get_coords(index[-1]))
        majeur_proj = projection_point(get_coords(majeur[0]), get_coords(majeur[1]), get_coords(majeur[-1]))
        distance_proj = np.linalg.norm(index_proj - majeur_proj)
        distance_base = np.linalg.norm(np.array(get_coords(index[0])) - np.array(get_coords(majeur[0])))
        ratio = -distance_proj / distance_base

        # Calcul si pouce croise index
        index_base_proj = projection_point(get_coords(index[0]), get_coords(auriculaire[0]), get_coords(index[0]))
        index_proj = projection_point(get_coords(index[0]), get_coords(auriculaire[0]), get_coords(index[-1]))
        pouce_proj = projection_point(get_coords(index[0]), get_coords(auriculaire[0]), get_coords(pouce[-1]))
        pouce_ind_croise = int(np.linalg.norm(index_base_proj - index_proj) > 
                              np.linalg.norm(index_base_proj - pouce_proj))

        # Calcul de l'angle entre index et majeur
        index_vector = np.array(get_coords(index[-1])) - np.array(get_coords(index[0]))
        majeur_vector = np.array(get_coords(majeur[-1])) - np.array(get_coords(majeur[0]))
        dot_product = np.dot(index_vector, majeur_vector)
        cos_theta = dot_product / (np.linalg.norm(index_vector) * np.linalg.norm(majeur_vector))
        angle_index_majeur = np.arccos(np.clip(cos_theta, -1.0, 1.0)) / np.pi

        return angles + [paume_vers_avant, majeur_sur_index, ratio, pouce_ind_croise, angle_index_majeur]

    def predict(self, img):
        # Conversion RGB format
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process avec mediapipe
        detection = self.detector.process(rgb_img)
        
        # Detection de la main
        if not detection.multi_hand_landmarks:
            return None
            
        #Detection droite gauche
        detection_obj = type('Detection', (), {})()
        detection_obj.hand_landmarks = [[]]
        detection_obj.handedness = [[type('Handedness', (), {'category_name': detection.multi_handedness[0].classification[0].label})()]]
        
        # Copie des landmarks
        detection_obj.hand_landmarks[0] = detection.multi_hand_landmarks[0].landmark
        
        # Mirroir si main gauche
        if detection.multi_handedness[0].classification[0].label == "Left":
            for landmark in detection_obj.hand_landmarks[0]:
                landmark.x = 1 - landmark.x

        # Preprocess  detection
        try:
            preprocessed = self.preprocess(detection_obj)
            preprocessed = np.array(preprocessed).reshape(1, -1)
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None

        try:
            # Model 1 prediction
            data_model_1 = self.scaler_1.transform(preprocessed)
            y_pred_1 = self.model_1.predict_proba(data_model_1)
            entropy_1 = -np.sum(np.clip(y_pred_1[0], 1e-10, 1) * np.log(np.clip(y_pred_1[0], 1e-10, 1))) / np.log(len(y_pred_1[0]))
            if entropy_1 > 0.8:
                return None

            # Model 2 prediction
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in detection.multi_hand_landmarks[0].landmark]).flatten()
            landmarks = landmarks.reshape(1, -1)
            data_model_2 = self.scaler_2.transform(landmarks)
            y_pred_2 = self.model_2.predict_proba(data_model_2)
            entropy_2 = -np.sum(np.clip(y_pred_2[0], 1e-10, 1) * np.log(np.clip(y_pred_2[0], 1e-10, 1))) / np.log(len(y_pred_2[0]))
            if entropy_2 > 0.9:
                return None

            # Combined model prediction
            data_model_3 = np.concatenate((y_pred_1, y_pred_2), axis=1)
            prediction = self.model_3.predict(data_model_3)[0]
            
            return chr(65 + prediction)
        except Exception as e:
            print(f"Error in model prediction: {e}")
            return None

def main():
    # Initalisation du prédicteur
    sign_predictor = SignPredictor()
    
    #Webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Prediction de l'image
        prediction = sign_predictor.predict(frame)
        
        # Si prédiction directement sur l'image
        if prediction:
            # Draw text in the top-left corner
            cv2.putText(frame, f'Sign: {prediction}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        # Show frame
        cv2.imshow('Sign Language Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()