import os

os.chdir('/home/pallab/Pictures/pos')
# os.chdir('/home/pallab/Videos')
i = 0
for pic in os.listdir('.'):
    if pic.startswith('im'):
        os.rename(pic, 'img'+str(i)+'.jpg')
        i += 1

print('Renamed',i, 'files', sep = ' ', end = '\n')
