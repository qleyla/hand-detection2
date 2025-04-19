import cv2
import mediapipe as mp
import pyautogui

# MediaPipe əl tanıma modulu
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            # Məsələn, baş barmaqla şəhadət barmağı bir-birinə yaxınsa, klik əmri ver
            if lm_list:
                x1, y1 = lm_list[4]   # baş barmaq
                x2, y2 = lm_list[8]   # şəhadət barmağı
                distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5

                if distance < 40:
                    pyautogui.click()
                    print("Klik edildi")

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
