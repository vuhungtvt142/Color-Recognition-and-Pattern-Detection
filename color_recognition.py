import cv2
import numpy as np

def nothing(x):
    pass

def get_color_name(a,b=0):
    # Ánh xạ các giá trị HSV vào tên màu
    color_dict = {
        (0, 11): 'Đỏ',         #0-10
        (10, 21): 'Cam',       #11-20
        (20, 41): 'Vàng',      #21-40
        (40, 81): 'Lục',       #31-80
        (80, 131): 'Xanh lam', #80-130
        (130, 161): 'Chàm',    #130-160
        (160, 180): 'Đỏ'       #160-179
    }

    for key, value in color_dict.items():
        #if (key[0] <= a < key[1]) and(key[0] <= b < key[1]):
        if key[0] <= a < key[1]:
            return value

def bgr_to_hex(bgr_color):
    # Chuyển đổi giá trị màu BGR sang mã hex
    hex_color = "#{:02x}{:02x}{:02x}".format(bgr_color[2], bgr_color[1], bgr_color[0])
    return hex_color.upper()  # Chuyển đổi sang chữ in hoa (tuỳ chọn)

# Load image
image = cv2.resize(cv2.imread(r'data/demo.png'),(500,500))

# Create a window
cv2.namedWindow('image')

# Create trackbars for color change
# Hue is from 0-179 for Opencv
cv2.createTrackbar('Hue Min', 'image', 0, 179, nothing)
cv2.createTrackbar('Saturation Min', 'image', 0, 255, nothing)
cv2.createTrackbar('Value Min', 'image', 0, 255, nothing)
cv2.createTrackbar('Hue Max', 'image', 0, 179, nothing)
cv2.createTrackbar('Saturation Max', 'image', 0, 255, nothing)
cv2.createTrackbar('Value Max', 'image', 0, 255, nothing)

# Set default value for Max HSV trackbars
cv2.setTrackbarPos('Hue Max', 'image', 179)
cv2.setTrackbarPos('Saturation Max', 'image', 255)
cv2.setTrackbarPos('Value Max', 'image', 255)

# Initialize HSV min/max values
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

while(1):
    image_copy=image.copy()

    # Get current positions of all trackbars
    hMin = cv2.getTrackbarPos('Hue Min', 'image')
    sMin = cv2.getTrackbarPos('Saturation Min', 'image')
    vMin = cv2.getTrackbarPos('Value Min', 'image')
    hMax = cv2.getTrackbarPos('Hue Max', 'image')
    sMax = cv2.getTrackbarPos('Saturation Max', 'image')
    vMax = cv2.getTrackbarPos('Value Max', 'image')

    # Set minimum and maximum HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Convert to HSV format and color threshold
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    #mask = cv2.dilate(mask, np.ones((5, 5), np.uint8) , iterations=1)
    result = cv2.bitwise_and(image, image, mask=mask)


    # Print if there is a change in HSV value
    if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        #print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # Display result image
    image_3d = np.tile(mask[:, :, np.newaxis], (1, 1, 3))
    side_by_side = np.hstack([result,image_3d])
    cv2.imwrite('image_color.png',result)
    contours, hierarchy = cv2.findContours(cv2.dilate(mask, np.ones((5, 5), np.uint8) , iterations=1) ,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    image_copy=cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 3)


    side_by_side = np.hstack([side_by_side,cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 3)])

    cv2.imshow('image',side_by_side)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


cluster_info = []

for i, contour in enumerate(contours):
    # Tính diện tích của contour (số pixel)
    area = cv2.contourArea(contour)

    # Tính giá trị trung bình của các pixel trong contour
    # mask = np.zeros_like(gray)
    # cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    mean_value = cv2.mean(hsv, mask=mask)[:3]

    cluster_info.append({
        'Cluster': i + 1,
        'Pixel Count': area,
        'Mean Value': mean_value,
        'BGR': cv2.cvtColor(np.array([[list(mean_value)]], dtype=np.uint8) , cv2.COLOR_HSV2BGR),
        'HEX': bgr_to_hex(tuple(cv2.cvtColor(np.array([[list(mean_value)]], dtype=np.uint8) , cv2.COLOR_HSV2BGR)[0][0]))
    })

# In thông tin về số pixel và giá trị các pixel trong từng cụm
for info in cluster_info:
    print(f"Cụm {info['Cluster']}: Số pixel = {info['Pixel Count']}, Giá trị trung bình = {info['Mean Value']}, BGR = {info['BGR']}, HEX= {info['HEX']}")

# Tổng số pixel trong tất cả các cụm
total_pixels = sum(info['Pixel Count'] for info in cluster_info)
print(f"Tổng số pixel: {total_pixels}")

print(get_color_name(info['Mean Value'][0]))

cv2.destroyAllWindows()
