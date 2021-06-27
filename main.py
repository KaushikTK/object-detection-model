
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
# evaluate the mask rcnn model on the kangaroo dataset
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from matplotlib import pyplot
from matplotlib.patches import Rectangle

class DATASET(Dataset):
    # this class show have 3 member functions - load_dataset, load_mask and 
    def load_dataset(self, dataset_dir, is_train=True):
        # define one class
        self.add_class(source="dataset", class_id=1, class_name="kangaroo") # start class_id with 1 => 0 is reserved for background

        # define data locations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'

        # find all images
        for filename in listdir(images_dir):
            # extract image id
            image_id = filename[:-4]

            if image_id in ['00090']:continue # image 90 has some issue so skip it

            # skip all images after 150 if we are building the train set
            if is_train and int(image_id) >= 150:continue

            # skip all images before 150 if we are building the test/val set
            if not is_train and int(image_id) < 150:continue

            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'

            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # function to extract data from xml
    def extractData(self, filename):
        tree = ElementTree.parse(filename) # read the xml file
        root = tree.getroot() # get the root of the document
        boxes = list()
        for box in root.findall('.//bndbox'):
            # extract the limits of the bounding box
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)

            boxes.append([xmin, ymin, xmax, ymax])

        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)

        return boxes, width, height

    # load the masks for an image
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']

        # extract data from the XML file
        boxes, w, h = self.extractData(path)
        
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]

            masks[row_s:row_e, col_s:col_e, i] = 1 # this makes all the pixels inside the bounding box as white and background as 0

            class_ids.append(self.class_names.index('kangaroo'))

        return masks, asarray(class_ids, dtype='int32')

    # load reference to image
    def imageReference(self, image_id):
        return self.image_info[image_id]['path']

# preparing the train and test dataset
trainSet = DATASET()
trainSet.load_dataset('kangaroo', is_train=True)
trainSet.prepare()
print('Train set: ' + str(len(trainSet.image_ids)))

testSet = DATASET()
testSet.load_dataset('kangaroo', is_train=False)
testSet.prepare()
print('Test set: ' + str(len(testSet.image_ids)))


#TRAIN

class CustomModelConfiguration(Config):
    NAME = 'OUR_CUSTOM_CONFIG' # name of the configuration
    NUM_CLASSES = 2 # kangaroo + background
    STEPS_PER_EPOCH = 131 # number of training steps

config = CustomModelConfiguration()
config.display()

# build the model
model = MaskRCNN(mode='training', model_dir='./', config=config)
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# train the model
model.train(trainSet, testSet, config.LEARNING_RATE, 5, 'heads')
model.save('trainedModel.h5')



# TEST

class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "kangaroo_cfg"
    # number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
    APs = list()
    for image_id in dataset.image_ids:
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)
        # extract results for first sample
        r = yhat[0]
        # calculate statistics, including AP
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        # store
        APs.append(AP)
    # calculate the mean AP across all images
    mAP = mean(APs)
    return mAP

# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights
model.load_weights('our_custom_config20210627T0013/mask_rcnn_our_custom_config_0005.h5', by_name=True)
# evaluate model on training dataset
train_mAP = evaluate_model(trainSet, model, cfg)
print("Train mAP: %.3f" % train_mAP)
# evaluate model on test dataset
test_mAP = evaluate_model(testSet, model, cfg)
print("Test mAP: %.3f" % test_mAP)



# PLOT
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "kangaroo_cfg"
    # number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# plot a number of photos with ground truth and predictions
def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
    # load image and mask
    for i in range(n_images):
        # load the image and mask
        image = dataset.load_image(i)
        mask, _ = dataset.load_mask(i)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)[0]
        # define subplot
        pyplot.subplot(n_images, 2, i*2+1)
        # plot raw pixel data
        pyplot.imshow(image)
        pyplot.title('Actual')
        # plot masks
        for j in range(mask.shape[2]):
            pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
        # get the context for drawing boxes
        pyplot.subplot(n_images, 2, i*2+2)
        # plot raw pixel data
        pyplot.imshow(image)
        pyplot.title('Predicted')
        ax = pyplot.gca()
        # plot each box
        for box in yhat['rois']:
            # get coordinates
            y1, x1, y2, x2 = box
            # calculate width and height of the box
            width, height = x2 - x1, y2 - y1
            # create the shape
            rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            # draw the box
            ax.add_patch(rect)
    # show the figure
    pyplot.show()

# load the train dataset
train_set = DATASET()
train_set.load_dataset('kangaroo', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# load the test dataset
test_set = DATASET()
test_set.load_dataset('kangaroo', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights
model_path = 'our_custom_config20210627T0013/mask_rcnn_our_custom_config_0005.h5'
model.load_weights(model_path, by_name=True)
# plot predictions for train dataset
plot_actual_vs_predicted(train_set, model, cfg)
# plot predictions for test dataset
plot_actual_vs_predicted(test_set, model, cfg)