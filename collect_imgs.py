import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Kamera initialisieren
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Fehler: Kamera konnte nicht geöffnet werden.")
    exit()

while True: # Schleife, um Daten für mehrere Klassen nacheinander sammeln zu können
    class_label = input("Für welche Klasse möchtest du Daten sammeln? (z.B. 0, 1, A ... oder 'ende' zum Beenden): ")
    if class_label.lower() == 'ende':
        break

    class_path = os.path.join(DATA_DIR, class_label)
    if not os.path.exists(class_path):
        os.makedirs(class_path)
        print(f"Ordner für Klasse '{class_label}' wurde erstellt: {class_path}")

    while True:
        try:
            num_images_to_collect_str = input(f"Wie viele Bilder möchtest du für Klasse '{class_label}' sammeln? ")
            num_images_to_collect = int(num_images_to_collect_str)
            if num_images_to_collect <= 0:
                print("Bitte eine positive Zahl eingeben.")
            else:
                break
        except ValueError:
            print("Ungültige Eingabe. Bitte eine Zahl eingeben.")


    # Bestimme den Startzähler für Dateinamen basierend auf existierenden Dateien
    try:
        existing_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # Extrahiere Zahlen aus Dateinamen wie "123.jpg"
        # Dies ist eine robustere Methode, um die höchste existierende Nummer zu finden
        highest_num = -1
        for f_name in existing_files:
            try:
                num = int(os.path.splitext(f_name)[0]) # Extrahiert "123" aus "123.jpg"
                if num > highest_num:
                    highest_num = num
            except ValueError:
                # Ignoriere Dateien, die nicht dem Zahlenformat entsprechen
                continue
        start_counter = highest_num + 1
    except Exception as e:
        print(f"Fehler beim Lesen existierender Dateien für Klasse {class_label}: {e}")
        start_counter = 0 # Fallback

    print(f"Sammle Daten für Klasse '{class_label}'. Existierende Bilder: {start_counter}. Starte mit '{start_counter}.jpg'.")
    print('Bereit? Zeige die Geste und drücke dann "Q", um mit dem Sammeln zu beginnen!')
    print('Drücke "ESC" während des Sammelns, um vorzeitig zu stoppen.')

    # Warte auf 'q' zum Starten
    display_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    display_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fehler: Frame konnte nicht von der Kamera gelesen werden.")
            break 
        
        # Position des Textes anpassen, falls die Auflösung klein ist
        text_y_pos = 50 if display_frame_height > 100 else 20
        font_scale = 1.3 if display_frame_width > 600 else 0.7

        cv2.putText(frame, 'Bereit? Zeige Geste, dann "Q" druecken!', (int(display_frame_width * 0.1), text_y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break
    
    if not ret: 
        continue 

    collected_count_this_session = 0
    for i in range(num_images_to_collect):
        ret, frame = cap.read()
        if not ret:
            print("Fehler: Frame konnte während des Sammelns nicht gelesen werden.")
            break
        cv2.imshow('frame', frame) 

        key = cv2.waitKey(100) # 100ms Pause, ca. 10 Bilder pro Sekunde maximal

        img_name = os.path.join(class_path, f'{start_counter + collected_count_this_session}.jpg')
        cv2.imwrite(img_name, frame)
        print(f"Gespeichert: {img_name}")

        collected_count_this_session += 1

        if key == 27: # ESC-Taste
            print("Sammeln vorzeitig durch Benutzer abgebrochen.")
            break
    
    print(f"{collected_count_this_session} Bilder für Klasse '{class_label}' in dieser Sitzung gesammelt.")
    print("-" * 30)

# Aufräumen
print("Datensammlung beendet.")
cap.release()
cv2.destroyAllWindows()