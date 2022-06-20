import cv2
import time
from datetime import datetime
import pyautogui

# setting
LOOP_SECONDS = 0.5 # 검사 루프의 간격
HIDE_LAYER_WHEN_NONE_PEOPLE = 14 # 몇번 루프에서 사람들이 없으면 레이어를 감출까요?
TAKE_CAPTURE_BETWEEN_LOOP = 40 # 사람 간격으로 20번 루프 이후에 다시 캡쳐 작업

video_capture = cv2.VideoCapture(0)
process_this_frame = True
is_layer_on = False
count_loop = 0
count_empty = 0
prev_capture_loop = 0

if not video_capture.isOpened():
    print("Camera open failed!")
    exit()

while True:
    count_loop += 1
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        grayImage = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        body_cascade = cv2.CascadeClassifier('./openCV_XML/haarcascade_upperbody.xml')
        body = body_cascade.detectMultiScale(grayImage, 1.02, 10)

    process_this_frame = not process_this_frame

    for (x, y, w, h) in body:

        x *= 4
        y *= 4
        w *= 4
        h *= 4

        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3) # 사람 인식하는 선 그리는 코드

        body_image_gray = grayImage[y:y + h, x:x + w]
        body_image_color = small_frame[y:y + h, x:x + w]

    if len(body) != 0:
        count_empty = 0
        if not is_layer_on:
            pyautogui.hotkey('ctrl', 'shift', 'q')
            is_layer_on = True
            print('layer: on')
        if count_loop - prev_capture_loop > TAKE_CAPTURE_BETWEEN_LOOP:
            print('capture:  because count_loop : ' + str(count_loop) +
                  ', prev_capture_loop: ' + str(prev_capture_loop))
            print('캡쳐되었습니다')
            cv2.imwrite('photos/' + datetime.now().strftime('%Y-%m-%d-%H.%M.%S') + '.jpg', frame)
            prev_capture_loop = count_loop
        else:
            print('not yet to take capture because count_loop : ' + str(count_loop) +
                  ', prev_capture_loop: ' + str(prev_capture_loop))
    else:
        count_empty += 1
        if count_empty > HIDE_LAYER_WHEN_NONE_PEOPLE and is_layer_on is True:
            pyautogui.hotkey('ctrl', 'shift', 'w')
            is_layer_on = False
            print('layer: off because count_empty : ' + str(count_empty))

    cv2.imshow('Video', frame)

    print('info: count_loop: ' + str(count_loop) + ', prev_capture_loop: ' +
          str(prev_capture_loop) + ', ' + str(len(body)) + 'bodies targeted')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("exit program")
        break

    time.sleep(LOOP_SECONDS)


video_capture.release()
cv2.destroyAllWindows()
