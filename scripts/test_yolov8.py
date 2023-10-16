from ultralyticsplus import YOLO, render_result
import os 
from tqdm import tqdm
# load model
model = YOLO('ultralyticsplus/yolov8s')

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

# set image
# image = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'

demo_folder = "banana_soup"

rgb_img_path = os.path.join("../demos/", demo_folder, "rgb")
save_path = os.path.join("../results/yolo", demo_folder)
os.makedirs(save_path)
for filename in tqdm(os.listdir(rgb_img_path)):
    image = os.path.join(rgb_img_path, filename)

    # perform inference
    results = model.predict(image)

    # observe results
    # print(results[0].boxes)
    render = render_result(model=model, image=image, result=results[0])

    render.save(os.path.join(save_path, filename))
    # render.show()
