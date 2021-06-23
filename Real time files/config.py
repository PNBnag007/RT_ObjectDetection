
# VOC_CLASSES = (  # always index 0
#     'aeroplane', 'bicycle', 'bird', 'boat',
#     'bottle', 'bus', 'car', 'cat', 'chair',
#     'cow', 'diningtable', 'dog', 'horse',
#     'motorbike', 'person', 'pottedplant',
#     'sheep', 'sofa', 'train', 'tvmonitor')

# VOC_CLASSES = (
#       'person','bicycle','car','motorcycle','airplane',
#     'bus','train','truck','boat','traffic','fire',
#     'street','stop','parking','bench','bird',
#     'cat','dog','horse','sheep','cow','elephant',
#     'bear','zebra','giraffe','hat','backpack','umbrella',
#     'shoe','eye','handbag','tie','suitcase','frisbee','skis',
#     'snowboard','sports','kite','baseball','baseball',
#     'skateboard','surfboard','tennis','bottle','plate',
#     'wine','cup','fork','knife','spoon','bowl',
#     'banana','apple','sandwich', 'orange','broccoli','carrot',
#     'hot','pizza','donut','cake','chair','couch',
#     'potted','bed','mirror','dining','window','desk',
#     'toilet','door','tv','laptop','mouse','remote','keyboard',
#     'cell','microwave','oven','toaster','sink','refrigerator',
#     'blender','book','clock','vase','scissors',
#     'teddy','hair','toothbrush','hair'
# )

VOC_CLASSES = ('person', 'bicycle','car','dog','horse','chair','laptop','mouse','keyboard','book')
VOC_IMG_MEAN = (123, 117, 104)  # RGB

COLORS = [[0, 0, 0],
         [128, 0, 0],
         [0, 128, 0],
         [128, 128, 0],
         [0, 0, 128],
         [128, 0, 128],
         [0, 128, 128],
         [128, 128, 128],
         [64, 0, 0],
         [192, 0, 0],
         [64, 128, 0],
         [192, 128, 0],
         [64, 0, 128],
         [192, 0, 128],
         [64, 128, 128],
         [192, 128, 128],
         [0, 64, 0],
         [128, 64, 0],
         [0, 192, 0],
         [128, 192, 0],
         [0, 64, 128]]

# network expects a square input of this dimension
YOLO_IMG_DIM = 448
