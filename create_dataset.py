import os
import pickle
import mediapipe as mp
import cv2
#
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

DATA_DIR = './data'  

data = []
labels = []

print(f"Starte Verarbeitung der Bilder aus dem Verzeichnis: {DATA_DIR}")


for class_label in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_label)


    if os.path.isdir(class_path):
        print(f"Verarbeite Bilder für Klasse: '{class_label}' aus Verzeichnis: {class_path}")
        
        for img_filename in os.listdir(class_path):
            img_full_path = os.path.join(class_path, img_filename)

            try:
              
                img = cv2.imread(img_full_path)

                if img is None:
                    print(f"Warnung: Bild konnte nicht geladen werden oder ist ungültig: {img_full_path}")
                    continue  

                # Bild von BGR zu RGB konvertieren (MediaPipe erwartet RGB)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Bild mit MediaPipe Hands verarbeiten, um Landmarken zu finden
                results = hands.process(img_rgb)

                # --- START DER MODIFIZIERTEN FEATURE-EXTRAKTION ---
                EXPECTED_COORDS_PER_HAND = 21 * 2  # 21 Landmarken * 2 Koordinaten (x, y)
                TOTAL_EXPECTED_FEATURES = EXPECTED_COORDS_PER_HAND * 2 # Für zwei Hände

           
                image_features = [0.0] * TOTAL_EXPECTED_FEATURES
                hands_actually_processed = 0 

                if results.multi_hand_landmarks:
                    
                    for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        if hand_idx >= 2:  
                            break

                        current_hand_xs = []
                        current_hand_ys = []
                        for landmark in hand_landmarks.landmark:
                            current_hand_xs.append(landmark.x)
                            current_hand_ys.append(landmark.y)
                        
                       
                        if current_hand_xs and current_hand_ys: 
                            min_x = min(current_hand_xs)
                            min_y = min(current_hand_ys)

                           
                            start_offset_in_features = hand_idx * EXPECTED_COORDS_PER_HAND
                            
                           
                            for i in range(len(current_hand_xs)): 
                                # x-Koordinate
                                image_features[start_offset_in_features + (i * 2)] = current_hand_xs[i] - min_x
                                # y-Koordinate
                                image_features[start_offset_in_features + (i * 2) + 1] = current_hand_ys[i] - min_y
                            
                            hands_actually_processed += 1
                
                if hands_actually_processed > 0:
                    data.append(image_features) 
                    labels.append(class_label)
         

            except Exception as e:
                print(f"FEHLER bei der Verarbeitung von Bild {img_full_path}: {e}")
                continue 
    else:
        if class_label != '.DS_Store': # Unterdrücke die Meldung für .DS_Store
            print(f"Info: Überspringe Eintrag (kein Verzeichnis): {class_path}")


if data and labels: 
    print(f"Speichere {len(data)} Datensätze...")
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print("Dataset-Erstellung abgeschlossen. Gespeichert in 'data.pickle'.")
else:
    print("Keine Daten zum Speichern vorhanden. 'data.pickle' wurde nicht erstellt.")
    # Verbesserte Überprüfung für leeres DATA_DIR oder fehlende Klassen-Unterverzeichnisse
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR) or \
       all(not os.path.isdir(os.path.join(DATA_DIR, item)) or item == '.DS_Store' for item in os.listdir(DATA_DIR)):
        print(f"Hinweis: Das Verzeichnis '{DATA_DIR}' scheint leer zu sein, existiert nicht oder enthält keine gültigen Klassen-Unterverzeichnisse.")
        print("Bitte stelle sicher, dass du mit 'collect_imgs.py' zuerst Bilder in Unterverzeichnissen (z.B. './data/0/', './data/1/') gesammelt hast.")

        