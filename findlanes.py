import cv2 as cv
import numpy as np

def make_coordinates(image, line_parameters):
    try:
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1 * (3 / 5))
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])
    except TypeError:
        return None


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return np.array([])

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_line = make_coordinates(image, np.average(left_fit, axis=0)) if left_fit else None
    right_line = make_coordinates(image, np.average(right_fit, axis=0)) if right_fit else None

    # Filter out None values
    averaged_lines = [line for line in [left_line, right_line] if line is not None]
    return np.array(averaged_lines)
    # print(left_fit_average, 'left')
    # print(right_fit_average, 'right')


def canny(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5),0)
    canny = cv.Canny(blur, 50, 150)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line
                cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv.fillPoly(mask, polygons, 255)
    masked_image = cv.bitwise_and(image, mask)
    return masked_image

#image = cv2.imread()
vidcap = cv.VideoCapture("highway45.mp4")
#success, image = vidcap.read()
while(vidcap.isOpened()):
    ret, frame = vidcap.read()
    if not ret:
        break

    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=80)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv.addWeighted(frame, 0.8, line_image, 1, 1)
    cv.imshow("result", combo_image)
    if cv.waitKey(100) == ord('q'):
        break

vidcap.release()
cv.destroyAllWindows()
