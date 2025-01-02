import sys
from pathlib import Path

# Add src/yolov7 to the Python path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[0]
sys.path.append(str(root / 'src' / 'yolov7'))

import cv2
import torch
import numpy as np
import mediapipe as mp
import joblib
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device
from models.experimental import attempt_load

class SignPredictor:
    def __init__(self):
        # Load the trained models and scalers
        self.scaler_1 = joblib.load("scaler_1.pkl")
        self.scaler_2 = joblib.load("scaler_2.pkl")
        self.model_1 = joblib.load("best_model_1.pkl")
        self.model_2 = joblib.load("best_model_2.pkl")
        self.model_3 = joblib.load("best_model_3.pkl")
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
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
        angles = get_angles(base_poignet, pouce, hand) + get_angles(base_poignet, index, hand) + \
                get_angles(base_poignet, majeur, hand) + get_angles(base_poignet, annulaire, hand) + \
                get_angles(base_poignet, auriculaire, hand)

        hand_without_fingers = np.array([get_coords(base_poignet), get_coords(pouce[0]), 
                                       get_coords(index[0]), get_coords(majeur[0]), 
                                       get_coords(annulaire[0]), get_coords(auriculaire[0])])
        
        U, S, Vt = np.linalg.svd(hand_without_fingers - np.mean(hand_without_fingers, axis=0))
        normal_vector = Vt[2] if np.dot(Vt[2], [1, 1, 1]) >= -np.dot(Vt[2], [1, 1, 1]) else -Vt[2]

        paume_vers_avant = 0
        mean_fingertips = np.mean([get_coords(pouce[-1]), get_coords(index[-1]), 
                                 get_coords(majeur[-1]), get_coords(annulaire[-1]), 
                                 get_coords(auriculaire[-1])], axis=0)
        
        if signed_distance(mean_fingertips, normal_vector, 
                         -np.dot(normal_vector, np.mean(hand_without_fingers, axis=0)), index) > 0:
            normal_vector = -normal_vector
            paume_vers_avant = 1

        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        a, b, c = normal_vector
        d = -np.dot(normal_vector, np.mean(hand_without_fingers, axis=0))

        majeur_sur_index = signed_distance(get_coords(majeur[-1]), normal_vector, d, index) - \
                          signed_distance(get_coords(index[-1]), normal_vector, d, index)

        index_proj = projection_point(get_coords(index[0]), get_coords(index[1]), get_coords(index[-1]))
        majeur_proj = projection_point(get_coords(majeur[0]), get_coords(majeur[1]), get_coords(majeur[-1]))
        distance_proj = np.linalg.norm(index_proj - majeur_proj)
        distance_base = np.linalg.norm(np.array(get_coords(index[0])) - np.array(get_coords(majeur[0])))
        ratio = -distance_proj / distance_base

        index_base_proj = projection_point(get_coords(index[0]), get_coords(auriculaire[0]), get_coords(index[0]))
        index_proj = projection_point(get_coords(index[0]), get_coords(auriculaire[0]), get_coords(index[-1]))
        pouce_proj = projection_point(get_coords(index[0]), get_coords(auriculaire[0]), get_coords(pouce[-1]))
        pouce_ind_croise = int(np.linalg.norm(index_base_proj - index_proj) > 
                              np.linalg.norm(index_base_proj - pouce_proj))

        index_vector = np.array(get_coords(index[-1])) - np.array(get_coords(index[0]))
        majeur_vector = np.array(get_coords(majeur[-1])) - np.array(get_coords(majeur[0]))
        dot_product = np.dot(index_vector, majeur_vector)
        cos_theta = dot_product / (np.linalg.norm(index_vector) * np.linalg.norm(majeur_vector))
        angle_index_majeur = np.arccos(np.clip(cos_theta, -1.0, 1.0)) / np.pi

        return angles + [paume_vers_avant, majeur_sur_index, ratio, pouce_ind_croise, angle_index_majeur]

    def predict(self, img):
        # Convert the image to RGB format
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image with MediaPipe
        detection = self.detector.process(rgb_img)
        
        # Check if hand is detected
        if not detection.multi_hand_landmarks:
            return None
            
        # Create a compatible detection object for preprocess method
        detection_obj = type('Detection', (), {})()
        detection_obj.hand_landmarks = [[]]
        detection_obj.handedness = [[type('Handedness', (), {'category_name': detection.multi_handedness[0].classification[0].label})()]]
        
        # Copy landmarks
        detection_obj.hand_landmarks[0] = detection.multi_hand_landmarks[0].landmark
        
        # Handle left hand by mirroring
        if detection.multi_handedness[0].classification[0].label == "Left":
            for landmark in detection_obj.hand_landmarks[0]:
                landmark.x = 1 - landmark.x

        # Preprocess the detection
        try:
            preprocessed = self.preprocess(detection_obj)
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None

        # Model 1 prediction
        data_model_1 = self.scaler_1.transform([preprocessed])
        y_pred_1 = self.model_1.predict_proba(data_model_1)
        entropy_1 = -np.sum(np.clip(y_pred_1[0], 1e-10, 1) * np.log(np.clip(y_pred_1[0], 1e-10, 1))) / np.log(len(y_pred_1[0]))
        if entropy_1 > 0.8:
            return None

        # Model 2 prediction
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in detection.multi_hand_landmarks[0].landmark]).flatten()
        data_model_2 = self.scaler_2.transform([landmarks])
        y_pred_2 = self.model_2.predict_proba(data_model_2)
        entropy_2 = -np.sum(np.clip(y_pred_2[0], 1e-10, 1) * np.log(np.clip(y_pred_2[0], 1e-10, 1))) / np.log(len(y_pred_2[0]))
        if entropy_2 > 0.9:
            return None

        # Combined model prediction
        data_model_3 = np.concatenate((y_pred_1, y_pred_2), axis=1)
        prediction = self.model_3.predict(data_model_3)[0]
        
        return chr(65 + prediction)

def main():
    # Model settings for YOLOv7
    weights = 'best.pt'  # path to trained weights
    device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    conf_thres = 0.25
    iou_thres = 0.45
    img_size = 640
    
    # Load YOLO model
    model, stride, names, device = load_model(weights, device)
    
    # Initialize sign predictor
    sign_predictor = SignPredictor()
    
    # Set up webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # YOLO processing
        original_shape = frame.shape
        img = letterbox(frame, img_size, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        # YOLO inference
        with torch.no_grad():
            pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        
        # Process YOLO detections
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], original_shape).round()
                
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # Extract hand region
                    hand_img = frame[y1:y2, x1:x2]
                    
                    # Predict sign
                    if hand_img.size > 0:  # Check if region is valid
                        prediction = sign_predictor.predict(hand_img)
                        
                        # Draw bounding box and prediction
                        if prediction:
                            label = f'Sign: {prediction} {conf:.2f}'
                            plot_one_box(xyxy, frame, label=label, color=[0, 255, 0], line_thickness=2)
        
        # Show frame
        cv2.imshow('Sign Language Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def load_model(weights='yolov7-tiny.pt', device=''):
    device = select_device(device)
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    names = model.module.names if hasattr(model, 'module') else model.names
    return model, stride, names, device

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

if __name__ == '__main__':
    main()