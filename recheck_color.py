import cv2
import numpy as np

# Tạo một ảnh trắng
image = np.ones((300, 500, 3), dtype=np.uint8) * 255  # 3 kênh màu (BGR), mỗi kênh có giá trị 255 (trắng)

# Vẽ hình chữ nhật màu xanh lá cây (BGR: (0, 255, 0))
cv2.rectangle(image, (50, 50), (200, 150), (148, 100,  38), thickness=cv2.FILLED)

# Hiển thị ảnh
cv2.imshow('Colored Shapes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()