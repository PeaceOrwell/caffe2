from caffe2.python import core, workspace
import cv2 as cv
import numpy as np
import detect_tool as tool

with open("init_net.pb") as f:
    init_net = f.read()

with open("predict_net.pb") as f:
    predict_net = f.read()

p = workspace.Predictor(init_net, predict_net)

pic_name = "dog.jpg"
img = cv.imread("dog.jpg")
img_min = img.min()
img_max = img.max()
img = (img - img_min) * 1.0 / (img_max -img_min)
img_resize = cv.resize(img, (608, 608))
img = img_resize * (img_max - img_min) + img_min
img_resize = img_resize.swapaxes(1, 2)
img_resize = img_resize.swapaxes(0, 1)
img_resize = img_resize[(2, 1, 0), :, :]
img_resize = np.array([img_resize], dtype = np.float32)

input_data = []
with open("data.txt", "r") as f:
    for line in f:
        t = line.strip()
        input_data.append(float(t))

input_img = np.array(input_data, dtype=np.float32)
input_img = input_img.reshape([1, 3, 608, 608])
print input_img.shape, input_img.dtype
res = p.run([input_img])
print len(res)
#res = np.array(res)
res_np = [np.array(r) for r in res]
name_list = ['data', 'conv1', 'scale1_w', 'scale1', "conv2", 'region1']
idx = 0
for a in res_np:
    a_flatten = a.flatten()
    with open("dump_{}.txt".format(name_list[idx]), "w") as f:
        for line in a_flatten:
            f.write("{}\n".format(line))
    print "write data to file ", "dump_{}.txt".format(name_list[idx])
    idx += 1

boxes_of_each_grid = 5
classes = 80
thread = 0.45
biases = np.array([0.738768, 0.874946, 2.42204, 2.65704, 4.30971, 7.04493, 10.246, 4.59428, 12.6868, 11.8741])
boxes = tool.get_region_boxes(res_np[5][0], boxes_of_each_grid, classes, thread, biases)

for box in boxes:
    print box

tool.draw_image(pic_name, boxes=boxes, namelist_file='coco.names')
