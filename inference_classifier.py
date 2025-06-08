import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

EXPECTED_COORDS_PER_HAND = 21 * 2  # 21 Landmarken * 2 Koordinaten (x, y)
TOTAL_EXPECTED_FEATURES = EXPECTED_COORDS_PER_HAND * 2 # Für zwei Hände

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamerabild konnte nicht gelesen werden.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux_for_model = [0.0] * TOTAL_EXPECTED_FEATURES
    all_x_coords_for_bbox = []
    all_y_coords_for_bbox = []
    hands_actually_processed = 0
    predicted_character = "-" # Standardwert

    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if hand_idx >= 2: 
                break
            
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            current_hand_xs_model = []
            current_hand_ys_model = []
            
            for landmark in hand_landmarks.landmark:
                current_hand_xs_model.append(landmark.x)
                current_hand_ys_model.append(landmark.y)
                all_x_coords_for_bbox.append(landmark.x)
                all_y_coords_for_bbox.append(landmark.y)

            if current_hand_xs_model and current_hand_ys_model:
                min_x_hand = min(current_hand_xs_model)
                min_y_hand = min(current_hand_ys_model)
                start_offset_in_features = hand_idx * EXPECTED_COORDS_PER_HAND
                
                for i in range(len(current_hand_xs_model)): 
                    data_aux_for_model[start_offset_in_features + (i * 2)] = current_hand_xs_model[i] - min_x_hand
                    data_aux_for_model[start_offset_in_features + (i * 2) + 1] = current_hand_ys_model[i] - min_y_hand
                hands_actually_processed +=1
        
        if hands_actually_processed > 0:
            try:
                prediction_output = model.predict([np.asarray(data_aux_for_model)])
                
                # --- HINZUGEFÜGTE DEBUG-ZEILEN ---
                print("--- DEBUG PREDICTION ---")
                print(f"Typ von prediction_output: {type(prediction_output)}")
                print(f"Inhalt von prediction_output: {prediction_output}")
                print("--- ENDE DEBUG PREDICTION ---")
                # --- ENDE DEBUG-ZEILEN ---
                
                # Annahme: prediction_output[0] ist bereits der Buchstabe, z.B. 'A'
                predicted_character = str(prediction_output[0])

            except Exception as e:
                print(f"Fehler bei der Vorhersage oder Verarbeitung: {e}")
                predicted_character = "Error"

            if all_x_coords_for_bbox and all_y_coords_for_bbox:
                x1 = int(min(all_x_coords_for_bbox) * W) - 10
                y1 = int(min(all_y_coords_for_bbox) * H) - 10
                x2 = int(max(all_x_coords_for_bbox) * W) + 10 
                y2 = int(max(all_y_coords_for_bbox) * H) + 10

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(W, x2)
                y2 = min(H, y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()