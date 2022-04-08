import cv2
import mediapipe as mp
import time
import random
import numpy as np

mphands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
rows,cols = (26,21)
set_x = [[0 for i in range(cols)] for j in range(rows)]
set_y = [[0 for i in range(cols)] for j in range(rows)]
set_z = [[0 for i in range(cols)] for j in range(rows)]

nodes_x = [0]*21
nodes_y = [0]*21
nodes_z = [0]*21

# img2 = cv2.imread(path)
# cv2.imshow('image', img2)
# img2RGB = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#
# results2 = hands.process(img2RGB)
# if results2.multi_hand_landmarks:
#     for handLms in results2.multi_hand_landmarks:
#         mpDraw.draw_landmarks(img2, handLms, mphands.HAND_CONNECTIONS)
# cv2.imshow('image', img2)
IMAGE_FILES = ['images/SignletterA.jpg','images/SignletterB.jpg','images/SignletterC.jpg','images/SignletterD.jpg','images/SignletterE.jpg','images/SignletterF.jpg','images/SignletterG.jpg','images/SignletterH.jpg']
with mphands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
    for idx, file in enumerate(IMAGE_FILES):  # for each sample image in the set
        image =cv2.imread(file)
        # image = cv2.flip(img, 1)  # may have to flip image?
        results = hands.process(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        print('Handedness:', results.multi_handedness)  # check specific hand being used?
        set_number = ord(file[17]) - ord('A')  # file[17] is 'letter', ord() is character in ascii
        if results.multi_hand_world_landmarks:  # if hand is detected
            for hand_world_landmarks in results.multi_hand_world_landmarks:  # get "world" landmarks
                for nodal_id, lm in enumerate(hand_world_landmarks.landmark):
                    set_x[int(set_number)][int(nodal_id)] = round(lm.x, 4)  # registers image nodes
                    set_y[int(set_number)][int(nodal_id)] = round(lm.y, 4)
                    set_z[int(set_number)][int(nodal_id)] = round(lm.z, 4)
    #print(set_x[0])
    #print(set_x[1])
    #print(set_x[2])
            #mpDraw.plot_landmarks(hand_world_landmarks, mphands.HAND_CONNECTIONS, azimuth=5)

cap = cv2.VideoCapture(0)  # Webcam setup
with mphands.Hands(
      model_complexity=1,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as hands:
    while cap.isOpened():  # Camera is on
        success,img = cap.read()  # get image
        matches = [0]*26
        text = ["a"]*26
        scoreText = "a"
        if success:  # if not failing
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)  # process hand image
            if result.multi_hand_world_landmarks:  # world landmarks
                for hand_world_landmarks in result.multi_hand_world_landmarks:
                    for nodal_id, lm in enumerate(hand_world_landmarks.landmark):
                        nodes_x[int(nodal_id)] = lm.x  # get values for current hand position
                        nodes_y[int(nodal_id)] = lm.y
                        nodes_z[int(nodal_id)] = lm.z

            if result.multi_hand_landmarks:  # local landmarks
                for hand_landmarks in result.multi_hand_landmarks:
                    for nodal_id, lm in enumerate(hand_landmarks.landmark):
                        # mpDraw.draw_landmarks(img, hand_landmarks, mphands.HAND_CONNECTIONS) #Draws the hand mesh
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)  # scale to image
                        for i in range(len(IMAGE_FILES)):  # for all samples
                            if abs(nodes_x[nodal_id] - set_x[i][nodal_id]) <0.04 and abs(
                                    nodes_y[nodal_id] - set_y[i][nodal_id]) < 0.04 and abs(nodes_z[nodal_id] - set_z[i][nodal_id]) < 0.04:
                                matches[i] = matches[i] + 1  # add to matches list, we can draw matching nodes here.
                                #random.seed(i)
                                #cv2.circle(img, (cx, cy), 15, (random.random()*255, random.random()*255, random.random()*255), cv2.FILLED)
            for i in range(len(IMAGE_FILES)):  # for all samples
                text[i] = chr(ord('A')+i) + str(matches[i])  # create text for matches
                cv2.putText(img, text[i], (10, 80+i*20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 3)
                if matches[i] == 21:  # if all nodes match, print letter being shown
                    scoreText = chr(ord('A')+i)
                    cv2.putText(img, scoreText, (10,30), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
            cv2.imshow("Image", img)  # show final image
            cv2.waitKey(1)
                # if abs(lm.x - sample_x[id]) < 0.05 and abs(
                #   lm.y - sample_y[id]) < 0.05 and abs(lm.z - sample_z[id]) < 0.05:
                #     mpDraw.draw_landmarks(img, hand_world_landmarks, mphands.HAND_CONNECTIONS)
                #     cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)

            #mpDraw.plot_landmarks(hand_world_landmarks, mphands.HAND_CONNECTIONS, azimuth=5)
            #mpDraw.draw_landmarks(img, handLms, mphands.HAND_CONNECTIONS)



def distance(node1,node2):
    dist = np.sqrt(np.square(nodes_x[node1]-nodes_x[node2])+np.square(nodes_y[node1]-nodes_y[node2])+np.square(nodes_z[node1]-nodes_z[node2]))
    return dist
# while True:
#     success, img, = cap.read()
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = hands.process(imgRGB)
#
#
#     if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks: #get each hand and extract information for each
#             for id, lm in enumerate(handLms.landmark):
#                 #print(id,type(lm))
#                 nodes_x[int(id)] = lm.x
#                 nodes_y[int(id)] = lm.y
#                 nodes_z[int(id)] = lm.z
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 #print(id, cx, cy)
#                 # if id == 4 or id == 3 or id == 2 or id == 1: #thumb
#                 #     cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
#                 # if id == 8 or id == 7 or id == 6 or id == 5: #index
#                 #     cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
#                 # if id == 12 or id == 11 or id == 10 or id == 9: #middle
#                 #     cv2.circle(img, (cx, cy), 15, (255, 255, 0), cv2.FILLED)
#                 # if id == 16 or id == 15 or id == 14 or id == 13 : #ring
#                 #     cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
#                 # if id == 20 or id == 19 or id == 18 or id == 17 : #pinky
#                 #     cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
#             #do comparisons in space
#             thumb = distance(0, 4)/distance(0,2)
#             print("0,4", thumb)
#             index = distance(0,8)/distance(0,5)
#             print("0,8", index)
#             middle = distance(0, 12)/distance(0,9)
#             print("0,12", middle)
#             ring = distance(0, 16)/distance(0,13)
#             print("0,16", ring)
#             pinky = distance(0, 20)/distance(0,17)
#             print("0,20", pinky)
#             closeval = 1.0
#             openval = 1.5
#             if(index < closeval and middle < closeval and ring < closeval and pinky < closeval):
#                 print("fist")
#             if (index > openval and middle > openval and ring > openval and pinky > openval):
#                 print("open hand")
#             mpDraw.draw_landmarks(img, handLms, mphands.HAND_CONNECTIONS)
#
#     cTime = time.time()
#     fps = 1/(cTime - pTime)
#     pTime = cTime
#
#     cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
#
# cv2.imshow("Image", img)
# cv2.waitKey(1)
#     #print("HERE")