# Image and Scene Recognition — Week 1 Lab

## Environment
- Python version: 3.11.9
- PyTorch version: 2.14.0+cpu
- torchvision version: 0.29.0+cpu
- OpenCV version: 5.0.0
- CUDA/GPU availability: False (CPU only)

## Week 1
This week's lab:
- configured the Python environment (`isr-env` virtual environment)
- verified that PyTorch and OpenCV are installed and working
- ran a pretrained ResNet-18 ImageNet classifier on a sample image
- produced top-3 predictions for `week1/sample.jpg`

## Classifier Result
Running `python week1/classify.py` on `week1/sample.jpg` produced:

```
Top-3 predictions:
 Samoyed                   88.46%
 Arctic fox                 4.58%
 white wolf                 4.43%
```

## Reflection

**1. Classification vs. detection vs. segmentation vs. captioning**

Image classification (recognition) looks at the whole image and assigns it one label, without saying where anything is located — this is exactly what the ResNet-18 exercise does: it outputs "Samoyed" for the whole picture, not a box around the dog. Object detection goes further by finding where each object is (usually as a bounding box) and labeling each one, so it can handle images with multiple objects. Segmentation is even more precise: instead of a box, it labels every individual pixel as belonging to an object or the background, giving an exact outline of each object's shape. Image captioning is different from all three — it generates a full natural-language sentence describing the image (e.g. "a white dog standing in the snow") instead of producing labels or boxes at all. Since our ResNet-18 exercise only assigns one label to the entire image, it is a classic example of image classification, not detection, segmentation, or captioning.

**2. Real-world industries that benefit from this kind of computer vision pipeline**

- **Autonomous driving**: a vehicle's camera system needs to recognize road signs, traffic lights, and general scene types (e.g. "highway" vs. "urban street") to help the car understand its environment and adjust driving behavior.
- **Retail**: stores can use image classification to automatically sort or tag product photos by category (e.g. shoes vs. shirts vs. electronics) for online catalogs, saving employees from doing this manually for thousands of items.

**3. Going from "classify the whole image" to "find and label every object"**

To move from whole-image classification to finding and labeling every object in an image, the model needs to also predict *where* each object is, not just *what* the dominant object is. This means switching from a classification model like ResNet-18 to an object detection model — for example, Faster R-CNN or YOLO — which is trained to output a bounding box (location) plus a class label for every object it finds in the image, instead of a single label for the whole picture. In short: the model architecture and training data both need to change from "one image → one label" to "one image → multiple boxes + labels."

## AI Assistance Disclosure

AI Assistance: Claude was used to assist with environment setup guidance, code preparation, and troubleshooting for this lab. The code was reviewed and executed by the student.
