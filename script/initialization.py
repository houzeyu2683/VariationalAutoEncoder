
"圖片存放在某個資料夾，根據圖片所在的位置建立表，存起來方便使用，減少記憶體過多浪費。"

import pandas, os

folder = os.path.join(os.environ['HOME'], "Public/##Data##/CelebFacesAttribute/JPG/")
group = os.listdir(folder)
link = [os.path.join(folder, i) for i in group]
table = pandas.DataFrame({'image':group, "link":link, 'mode':"train"})
pass

table = table.sample(len(table)).reset_index(drop=True)
size = int(len(table) * 0.1)
table.loc[range(size), 'mode'] = 'test'
pass

record = os.path.join(os.environ['HOME'], "Public/##Data##/CelebFacesAttribute/CSV/")
os.makedirs(record, exist_ok=True)
table.to_csv(os.path.join(record, 'index.csv'), index=False)
pass


# train = pandas.read_csv("../#DATA#/SFDDD/CSV/ANNOTATION.csv")
# train['mode'] = 'train'
# train['folder'] = "../#DATA#/SFDDD/JPG/TRAIN/"

# test = pandas.DataFrame({
#     'mode':'test', 
#     "img":os.listdir("../#DATA#/SFDDD/JPG/TEST"), 
#     "classname":"", 
#     'folder':"../#DATA#/SFDDD/JPG/TEST/"
# })

# train['image'] = train['folder'] + train['classname'] + '/' + train['img']
# test['image'] = test['folder'] + test['classname'] + test['img']

# table = pandas.concat([train, test])
# target = {
#     "c0":0,
#     "c1":1,
#     "c2":2,
#     "c3":3,
#     "c4":4,
#     "c5":5,
#     "c6":6,
#     "c7":7,
#     "c8":8,
#     "c9":9,
#     "":-1
# }
# table['target'] = table['classname'].replace(target)

# path = "SOURCE/CSV/ANNOTATION.csv"
# os.makedirs(os.path.dirname(path), exist_ok=True)
# table.to_csv(path, index=False)


# import matplotlib.pyplot as plt

# from skimage.feature import hog
# from skimage import data, exposure


# image = data.astronaut()

# fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)


# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

# ax1.axis('off')
# ax1.imshow(image, cmap=plt.cm.gray)
# ax1.set_title('Input image')

# # Rescale histogram for better display
# hog_image
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
# PIL.Image.fromarray(hog_image).convert("RGB").save("demo.png")
# ax2.axis('off')
# ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
# ax2.set_title('Histogram of Oriented Gradients')
# plt.show()