import cv2
from aug_algorithms import brightness


def get_brightness(image_frame):
    hue_saturation_value = cv2.cvtColor(image_frame, cv2.COLOR_BGR2HSV)
    _, _, value = cv2.split(hue_saturation_value)
    return value


cap = cv2.VideoCapture('input_video.mp4')
ret, frame = cap.read()
size = (frame.shape[1], frame.shape[0])

no_action_video = cv2.VideoWriter(filename='frames_without_actions.mp4', fourcc=cv2.VideoWriter_fourcc(*'MP4V'),
                                  fps=20, frameSize=size)
croses_video = cv2.VideoWriter(filename='input_video_with_red_cross.mp4', fourcc=cv2.VideoWriter_fourcc(*'MP4V'),
                               fps=20, frameSize=size)

previous_frame = frame
no_action_video.write(frame)
croses_video.write(frame)
previous_frame_birghtness = get_brightness(frame)
flag = 0
while True:
    ret, current_frame = cap.read()
    current_brightness = get_brightness(current_frame)
    delta_brightness = cv2.absdiff(current_brightness, previous_frame_birghtness)

    # ! try enhancing the brightness the frames (not OK)
    # * try reducing the brightness the frames (OK)
    previous_frame_bright = brightness(previous_frame, param=10)
    current_frame_bright = brightness(current_frame, param=10)

    # ? try reducing the noise
    # current_frame_bright = cv2.fastNlMeansDenoisingColored(current_frame_bright,None,10,10,7,21)
    previous_gray = cv2.cvtColor(previous_frame_bright, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame_bright, cv2.COLOR_BGR2GRAY)

    # ! try blurring the frames
    previous_gray_blur = cv2.GaussianBlur(src=previous_gray, ksize=(21, 21), sigmaX=0)
    current_gray_blur = cv2.GaussianBlur(src=current_gray, ksize=(21, 21), sigmaX=0)

    # calculating difference between the 2 frames
    diff = cv2.absdiff(previous_gray_blur, current_gray_blur)

    # thresholding
    _, diff_blur_thresh = cv2.threshold(src=diff, thresh=20, maxval=255, type=cv2.THRESH_BINARY)
    diff_blur_thresh = cv2.dilate(src=diff_blur_thresh, kernel=None, iterations=2)

    counts, _ = cv2.findContours(image=diff_blur_thresh.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    max_area_size = -1
    for c in counts:
        M = cv2.moments(c)
        area = M['m00']
        if area > max_area_size:
            max_area_size = area

    display_frame = current_frame.copy()
    # print(max_area_size, len(counts))
    if max_area_size < 10 or max_area_size > 314000:

        cv2.line(img=display_frame, pt1=(0, 0), pt2=(current_frame.shape[1] - 1, current_frame.shape[0] - 1),
                 color=(0, 0, 255), thickness=5, lineType=cv2.LINE_4)
        cv2.line(img=display_frame, pt1=(current_frame.shape[1] - 1, 0), pt2=(0, current_frame.shape[0] - 1),
                 color=(0, 0, 255), thickness=5, lineType=cv2.LINE_4)
        if flag != 0:
            print(flag)
        flag = 0
    else:
        no_action_video.write(current_frame)
        flag += 1

    cv2.imshow('current_frame', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    croses_video.write(display_frame)

    previous_frame = current_frame
    previous_birghtness = current_brightness

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
