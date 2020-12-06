import onnxruntime as ort

import numpy as np

class NoGPUAvailable(Exception):
    pass

class Wrapper():
    def __init__(self, model_file):
        #Load the model
        self.sess_ort = ort.InferenceSession("/code/exercise_ws/checkpoints/segmentation.onnx")
        
    def predict(self, batch_or_image):
        # TODO Make your model predict here!

        # TODO The given batch_or_image parameter will be a numpy array (ie either a 224 x 224 x 3 image, or a
        # TODO batch_size x 224 x 224 x 3 batch of images)
        # TODO These images will be 224 x 224, but feel free to have any model, so long as it can handle these
        # TODO dimensions. You could resize the images before predicting, make your model dimension-agnostic somehow,
        # TODO etc.

        # TODO This method should return a tuple of three lists of numpy arrays. The first is the bounding boxes, the
        # TODO second is the corresponding labels, the third is the scores (the probabilities)
        #img=batch_or_image[0:120,0:160,:]
        img = np.random.random_sample((1,120,160,3)) * 255
        res = self.sess_ort.run(output_names=["output:0"], input_feed={"input_rgb:0": img.reshape((1,120,160,3)).astype(np.float32)})

        # See this pseudocode for inspiration
        boxes = []
        labels = []
        scores = []
        for img in batch_or_image:  # or simply pipe the whole batch to the model instead of using a loop!
            pass
            #box, label, score = self.model.predict(img) # TODO you probably need to send the image to a tensor, etc.
            #boxes.append(box)
            #labels.append(label)
            #scores.append(score)

        return boxes, labels, scores

class Model():    # TODO probably extend a TF or Pytorch class!
    def __init__(self):
        # TODO Instantiate your weights etc here!
        pass
    # TODO add your own functions if need be!
