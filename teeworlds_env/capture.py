import time

from mss import mss

mon = {'top': 160, 'left': 160, 'width': 200, 'height': 200}

sct = mss()

while 1:
    x = time.time()
    sct_img = sct.grab(mon)
    print(time.time() - x)
    #img = Image.frombytes('RGB', (sct_img.width, sct_img.height), sct_img.rgb)
    #cv2.imshow('test', np.array(img))
    #if cv2.waitKey() & 0xFF == ord('q'):
    #    cv2.destroyAllWindows()
    #    break