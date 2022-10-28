import cv2


def main():
    image = cv2.imread('./laptop.jpg')
    cv2.imshow('Laptop', image)
    cv2.waitKey(0)

    cv2.putText(image, 'David', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    cv2.putText(image, 'Luca', (175, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

    cv2.imshow('Laptop', image)
    cv2.waitKey(0)

    cv2.imwrite('laptop_with_name.jpg', image)


if __name__ == '__main__':
    main()
