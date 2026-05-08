import cv2
import time

print('OpenCV version:', cv2.__version__)
print('Default backend:', cv2.getBuildInformation().splitlines()[0])

# Try common Windows backends and indices
backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
backend_names = ['CAP_DSHOW', 'CAP_MSMF', 'CAP_ANY']

for bi, backend in enumerate(backends):
    print('\nTrying backend:', backend_names[bi])
    for i in range(5):
        cap = cv2.VideoCapture(i, backend)
        opened = cap.isOpened()
        print(f' index={i} opened={opened}')
        if opened:
            ret, frame = cap.read()
            print('  read frame:', ret)
            if ret and frame is not None:
                win = f'cam_{backend_names[bi]}_{i}'
                cv2.imshow(win, frame)
                cv2.waitKey(1000)
                cv2.destroyWindow(win)
        cap.release()
        time.sleep(0.1)

print('\nIf none opened:')
print('- Close other apps using the camera (Teams/Zoom/Chrome)')
print('- Try changing USB port or camera index in `PostureTest.py`')
print('- On Windows, allow apps to access camera: Settings -> Privacy -> Camera')
print('- If using WSL/Remote, GUI windows may not appear; run on native Windows Python')
