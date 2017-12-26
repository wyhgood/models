import os
import shutil

i = 0
for f in os.listdir('test_image/'):
    i +=1
    print(f)
    shutil.move('./test_image/'+f, './test_image/'+str(i)+'.jpg')
