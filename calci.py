import cv2
import numpy as np

cap = cv2.VideoCapture(0)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

lowergreen = np.array([50, 100, 50])
uppergreen = np.array([90, 255, 255])


def draw_rectangle_at_coordinates(x1, y1, x2, y2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 255), 2)


def get_roi(frame, x1, y1, x2, y2):
    roi = frame[y1: y2, x1: x2, :]
    return roi


def get_hsv_roi(roi):
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    return hsv_roi


def get_contours_and_heirarchy()

while cap.isOpened():
    ret, frame = cap.read()
    # flipping the frame
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (int(width), int(height)))
    draw_rectangle_at_coordinates(100, 100, 500, 300)
    draw_rectangle_at_coordinates(600, 100, 1000, 300)
    draw_rectangle_at_coordinates(100, 400, 500, 600)
    draw_rectangle_at_coordinates(600, 400, 1000, 600)
    roi_1 = get_roi(frame, 100, 100, 500, 300)
    roi_2 = get_roi(frame, 600, 100, 1000, 300)
    roi_3 = get_roi(frame, 100, 400, 500, 600)
    roi_4 = get_roi(frame, 600, 400, 1000, 600)
    hsv_roi_1 = get_hsv_roi(roi_1)
    hsv_roi_2 = get_hsv_roi(roi_2)
    hsv_roi_3 = get_hsv_roi(roi_3)
    hsv_roi_4 = get_hsv_roi(roi_4)
    contours_1, hierarchy_1 = get_contours_and_heirarchy(roi_1_range.copy())
    contours_1, hierarchy_2 = get_contours_and_heirarchy(roi_2)
    contours_1, hierarchy_3 = get_contours_and_heirarchy(roi_3)
    contours_1, hierarchy_4 = get_contours_and_heirarchy(roi_4)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
