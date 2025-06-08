import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
import os # Sicherstellen, dass os importiert ist

script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, 'model.p')

try:
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    model = model_dict['model']
except FileNotFoundError:
    st.error(f"FEHLER: Modelldatei nicht gefunden unter {model_path}")
    st.stop() #
except Exception as e:
    st.error(f"FEHLER beim Laden des Modells: {e}")
    st.stop()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

# Definieren der erwarteten Feature-Längen
EXPECTED_COORDS_PER_HAND = 21 * 2  
TOTAL_EXPECTED_FEATURES = EXPECTED_COORDS_PER_HAND * 2 

if 'erkannte_woerter' not in st.session_state:
    st.session_state.erkannte_woerter = []
if 'letztes_hinzugefuegtes_wort' not in st.session_state:
    st.session_state.letztes_hinzugefuegtes_wort = ""
if 'kamera_aktiv' not in st.session_state:
    st.session_state.kamera_aktiv = False # Startet nicht automatisch

# --- Streamlit UI Aufbau ---
st.set_page_config(layout="wide") 
st.title("Gebärdensprache zu Text Dashboard")

col1, col2 = st.columns([3, 2]) # Spalten für Layout

with col1:
    st.header("Kamera Feed")
    start_kamera_button = st.button("Kamera starten/stoppen")
    frame_placeholder = st.empty() # Platzhalter für das Kamerabild

with col2:
    st.header("Erkennung")
    aktuelle_prediction_placeholder = st.subheader("Aktuell erkannt: -")
    
    st.header("Gesammelte Wörter")
    satz_text_area = st.text_area("Sequenz:", value=" ".join(st.session_state.erkannte_woerter), height=200, key="satz_display")
    
    if st.button("Sequenz löschen"):
        st.session_state.erkannte_woerter = []
        st.session_state.letztes_hinzugefuegtes_wort = ""
        satz_text_area.text_area("Sequenz:", value="", height=200, key="satz_display_cleared")

if start_kamera_button:
    st.session_state.kamera_aktiv = not st.session_state.kamera_aktiv
    if not st.session_state.kamera_aktiv: 
        st.session_state.erkannte_woerter = []
        st.session_state.letztes_hinzugefuegtes_wort = ""


if st.session_state.kamera_aktiv:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Kamera konnte nicht geöffnet werden.")
        st.stop()

    while st.session_state.kamera_aktiv and cap.isOpened(): # Schleife nur, wenn Kamera aktiv sein soll
        ret, frame = cap.read()
        if not ret:
            st.warning("Kamerabild konnte nicht gelesen werden. Versuche erneut...")
            break 

        H, W, _ = frame.shape
        frame_fuer_zeichnung = frame.copy() # Kopie für Zeichnungen, Original für RGB-Konvertierung
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Für MediaPipe
        results = hands.process(frame_rgb)

        data_aux_for_model = [0.0] * TOTAL_EXPECTED_FEATURES
        all_x_coords_for_bbox = []
        all_y_coords_for_bbox = []
        hands_actually_processed = 0
        current_predicted_char = "-" # Standardwert für aktuelle Anzeige

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if hand_idx >= 2: 
                    break
                
                mp_drawing.draw_landmarks(
                    frame_fuer_zeichnung, hand_landmarks, mp_hands.HAND_CONNECTIONS,
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
                    
                    
                    current_predicted_char = str(prediction_output[0])

                    # Logik zum Hinzufügen zur Wortsequenz (Debouncing)
                    if current_predicted_char and current_predicted_char != "-" and current_predicted_char != "Error" and current_predicted_char != "Unbekannt":
                        if current_predicted_char != st.session_state.letztes_hinzugefuegtes_wort:
                            st.session_state.erkannte_woerter.append(current_predicted_char)
                            st.session_state.letztes_hinzugefuegtes_wort = current_predicted_char
                            # Erzwinge Neuzeichnen der Text Area, da Streamlit Änderungen in Listen in session_state nicht immer sofort erkennt für alle Widgets
                            satz_text_area.text_area("Sequenz:", value=" ".join(st.session_state.erkannte_woerter), height=200, key=f"satz_display_{len(st.session_state.erkannte_woerter)}")


                except Exception as e:
                    print(f"Fehler bei der Vorhersage oder Verarbeitung: {e}") 
                    current_predicted_char = "Error"

                if all_x_coords_for_bbox and all_y_coords_for_bbox:
                    x1 = int(min(all_x_coords_for_bbox) * W) - 10
                    y1 = int(min(all_y_coords_for_bbox) * H) - 10
                    x2 = int(max(all_x_coords_for_bbox) * W) + 10 
                    y2 = int(max(all_y_coords_for_bbox) * H) + 10

                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(W, x2); y2 = min(H, y2)

                    cv2.rectangle(frame_fuer_zeichnung, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame_fuer_zeichnung, current_predicted_char, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            else: # Keine Hände verarbeitet
                 current_predicted_char = "-" # Setze auf Standard, wenn keine Hand da war

        # UI Elemente aktualisieren
        aktuelle_prediction_placeholder.subheader(f"Aktuell erkannt: {current_predicted_char}")

        # Frame für Streamlit vorbereiten (RGB) und anzeigen
        frame_placeholder.image(cv2.cvtColor(frame_fuer_zeichnung, cv2.COLOR_BGR2RGB), channels="RGB")
        
    
    if cap.isOpened(): # Sicherstellen, dass cap freigegeben wird, wenn die Schleife endet
        cap.release()

elif 'kamera_aktiv' in st.session_state and not st.session_state.kamera_aktiv :
     frame_placeholder.empty() # Leert das Bild, wenn Kamera gestoppt ist
     aktuelle_prediction_placeholder.subheader("Aktuell erkannt: -")

# Um die App zu starten: streamlit run dein_skript_name.py