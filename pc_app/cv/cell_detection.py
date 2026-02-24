import cv2

def detect_cell_borders(frame, min_area=100, max_area=5000):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    output = frame.copy()
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            cv2.drawContours(output, [cnt], -1, (0, 255, 255), 1)
            count += 1

    return output, count