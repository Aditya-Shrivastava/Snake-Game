import cv2
import numpy as np
from time import time
import random
import math

#iinitialize font for puttext
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

#loading apple image
apple = cv2.imread('apple.png', -1)
#print (apple.shape)

#making its mask to overlay ono video
apple_mask = apple[:, : , 3]
apple_mask_inv = cv2.bitwise_not(apple_mask)
apple = apple[:, :, 0:3]

#resizing apple image
apple = cv2.resize(apple, (40, 40), interpolation = cv2.INTER_AREA)
apple_mask = cv2.resize(apple_mask, (40, 40), interpolation = cv2.INTER_AREA)
apple_mask_inv = cv2.resize(apple_mask_inv, (40, 40), interpolation = cv2.INTER_AREA)

#capture webcam
video = cv2.VideoCapture(0)

#morphological operations
kernel_erode = np.ones((4, 4), np.uint8)
kernel_close = np.ones((15, 15), np.uint8)

def detect_color(hsv):
     #for red color
     lower = np.array([136, 87, 111])
     upper = np.array([179, 255, 255])
     mask1 =cv2.inRange(hsv, lower, upper)
     lower = np.array([0, 110, 100])
     upper = np.array([3, 255, 255])
     mask2 = cv2.inRange(hsv, lower, upper)
     mask_col = mask1 + mask2
     #erosion
     mask_col = cv2.erode(mask_col, kernel_erode, iterations = 1)
     #closing
     mask_col = cv2.morphologyEx(mask_col, cv2.MORPH_CLOSE, kernel_close)
     return mask_col

#detecting intersections of line segments
def direction(a, b, c):
     val = int(((b[1] - a[1]) * (c[0] - b[0])) - ((b[0] - a[0]) * (c[1] - b[1])))
     if val == 0:
          #linear
          return 0
     elif val > 0:
          #clockwise
          return 1
     else:
          #anti-cloclwise
          return 2

def intersection(a, b, c, d):
     d1 = direction(a, b, c)
     d2 = direction(a, b, d)
     d3 = direction(c, d, a)
     d4 = direction(c, d, b)
     if(d1 != d2 and d3 != d4):
          return True

     return False

start_time = int(time())
q, snake_len, score, temp = 0, 100, 0, 1

# center point of head
point_x, point_y = 0, 0

# points which satisfy condition, distance between 2 consecutive points, length of snake
last_point_x, last_point_y, dist, length = 0, 0, 0, 0

#store all points on snakke body
points = []

#store length between all points
list_len = []

#random pos for apple image
random_x = random.randint(10, 550)
random_y = random.randint(10, 400)

#for checking intersections
p, q, r, s = [], [], [], []

while 1:
     xr, yr, wr, hr = 0, 0, 0, 0
     _, frame = video.read()

     #horizontal flipping
     frame = cv2.flip(frame, 1)

     #initializing the accepted points
     if(q == 0 and point_x != 0 and point_y != 0):
          last_point_x = point_x
          last_point_y = point_y
          q = 1

     #converting to hsv
     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     mask_col = detect_color(hsv)

     #fiinding contours
     _, contour_col, _ = cv2.findContours(mask_col, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

     #drawing rectangle around head
     try:
          for i in range (0, 10):
               xr, yr, wr, hr = cv2.boundingRect(contour_col[i])
               if(wr * hr) > 2000:
                    break

     except:
          pass
     cv2.rectangle(frame, (xr, yr), (xr + wr, yr + hr), (0, 0, 255), 2)

     #snake body
     point_x = int(xr+(wr/2))
     point_y = int(yr+(hr/2))

     #distance between last and first point
     dist = int(math.sqrt(pow((last_point_x - point_x), 2) + pow((last_point_y - point_y), 2)))
     if(point_x != 0 and point_y != 0 and dist > 5):
          #if a point is accepted it is added to points list and its length to list_len
          list_len.append(dist)
          length += dist
          last_point_x = point_x
          last_point_y = point_y
          points.append([point_x, point_y])

     #if length becomes greater than expected length, remove points from back
     if(length >= snake_len):
          for i in range (len(list_len)):
               length -= list_len[0]
               list_len.pop(0)
               points.pop(0)
               if(length <= snake_len):
                    break
     
     #initialize blank black image
     blank_img = np.zeros((480, 640, 3), np.uint8)

     #drawing line between all points
     for i, j  in enumerate(points):
          if(i == 0):
               continue
          cv2.line(blank_img, (points[i-1][0], points[i-1][1]), (j[0], j[1]), (0, 0, 255), 5)
     cv2.circle(blank_img, (last_point_x, last_point_y), 5, (10, 200, 150), -1)

     #if snake eats apple: increase score and find new pos for apple
     if(last_point_x > random_x and last_point_x < (random_x+40) and last_point_y > random_y and last_point_y < (random_y+40)):
          score += 1
          random_x = random.randint(10, 550)
          random_y = random.randint(10, 400)

     #adding blank img to frame
     frame = cv2.add(frame, blank_img)

     #adding apple to frame
     roi = frame[random_y:random_y+40, random_x:random_x+40]
     img_bg = cv2.bitwise_and(roi, roi, mask = apple_mask_inv)
     img_fg = cv2.bitwise_and(apple, apple, mask = apple_mask)
     dst = cv2.add(img_bg, img_fg)
     frame[random_y:random_y+40, random_x:random_x+40] = dst
     cv2.putText(frame, str("SCORE - " + str(score)), (250, 450), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

     #check for snake hitting itself
     if(len(points) > 5):
          #p and q are head points of snake and r and s are other points
          q = points[len(points) - 2]
          p = points[len(points) - 1]
          for i in range (len(points) - 3):
               r = points[i]
               s = points[i+1]
               if(intersection(p, q, r, s) and len(r) != 0 and len(s) != 0):
                    temp = 0
                    break
          if temp == 0:
               break
     
     
     cv2.imshow("SNAKE", frame)
     #increase length of snake by 40px per second
     if(int(time()) - start_time) > 1:
          snake_len += 40
          start_time = int(time())
     key = cv2.waitKey(1)
     if key == 27:
          break

video.release()
cv2.destroyAllWindows()
cv2.putText(frame, str("GAME OVER!"), (100, 230), font, 3, (255, 0, 0), 3, cv2.LINE_AA)
cv2.putText(frame, str("PRESS ANY KEY TO EXIT"), (180, 260), font, 3, (255, 200, 0), 2, cv2.LINE_AA)
cv2.imshow("SNAKE", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
