## Drone Detection

In war-torn regions, effective drone identification can be a lifesaver. Therefore, machine learning-based object detection frameworks are applied to recognize drones in images.


```python
import  PIL
import yaml
import torch
from torch.utils.data import DataLoader
import argparse
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
import seaborn as sns
from PIL import Image
import statistics
import os
import plotly.graph_objects as go
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
import numpy as np
import  cv2
from plotly.subplots import make_subplots
```

Parameter values and hyperlink addresses will be loaded from the configuration file:


```python
with open('cfg.yaml', 'r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)
```


```python
torch.cuda.is_available()
```




    True



A custom dataset has been assembled using images and COCO annotations sourced from https://universe.roboflow.com/aaaa-psk4d/sinek-avi on Roboflow. To manage data batches, the custom_collate_fn function was implemented.


```python
class Drone_Dataset(Dataset):
    def __init__(self, cfg_path, transform=None, train = True):
          
        with open(cfg_path, 'r', encoding='utf-8') as file:
            self.cfg = yaml.safe_load(file)
        self.transform = transforms.Compose([
            transforms.ToTensor()])
        self.data_path = self.cfg['data']['data_path']
        
        if train == True:
            self.img_dir = self.data_path + "train/"
        else:
            self.img_dir = self.data_path + "test/"

        annotations_path =self.img_dir + "_annotations.coco.json"
        self.coco = COCO(annotations_path)
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.image_ids = self.coco.getImgIds()
        self.images = self.coco.loadImgs(self.image_ids)


    def __len__(self):
        return len(self.image_ids )
    

    def __getitem__(self, idx):     
            img_id = self.image_ids[idx]
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.coco.getCatIds(), iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            image_info = self.coco.loadImgs(img_id)[0]
            image_path = os.path.join(self.img_dir, image_info['file_name'])
            image = Image.open(image_path)
            image = self.transform(image)
            boxes = []
            classes = []
            for ann in anns:
                x_min, y_min, width, height = ann['bbox']
                boxes.append([x_min, y_min, x_min + width, y_min + height])  
                classes.append(ann['category_id'])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            classes = torch.as_tensor(classes, dtype=torch.int64)
            return img_id, image, boxes, classes 
```


```python
def custom_collate_fn(batch):
        img_id =  [item[0] for item in batch]
        images = [item[1] for item in batch]  
        boxes = [item[2] for item in batch]  
        classes = [item[3] for item in batch] 
        images = torch.stack(images, dim=0)
        return img_id, images, boxes, classes
```


```python
output_path = cfg["data"]["output"]
num_classes = cfg["model"]["num_classes"] 
```


```python
batch_size=cfg["training"]["batch_size"]
dataset = Drone_Dataset("cfg.yaml")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,  collate_fn=custom_collate_fn)
```


```python
test_dataset = Drone_Dataset("cfg.yaml", train = False)
test_dataloader= DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0,  collate_fn=custom_collate_fn)
```

The FasterRCNN architecture, with 'resnet50' serving as the backbone, was chosen to address the object detection requirements. The model was initialized using the create_faster_rcnn_model function, which configures the backbone to be pretrained and non-trainable, ensuring that existing learned features are used.


```python
def create_faster_rcnn_model(num_classes):
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model
```


```python
model = create_faster_rcnn_model(num_classes)
```


```python
model
```




    FasterRCNN(
      (transform): GeneralizedRCNNTransform(
          Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
          Resize(min_size=(800,), max_size=1333, mode='bilinear')
      )
      (backbone): BackboneWithFPN(
        (body): IntermediateLayerGetter(
          (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
          (bn1): FrozenBatchNorm2d(64, eps=1e-05)
          (relu): ReLU(inplace=True)
          (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
          (layer1): Sequential(
            (0): Bottleneck(
              (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(64, eps=1e-05)
              (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(64, eps=1e-05)
              (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(256, eps=1e-05)
              (relu): ReLU(inplace=True)
              (downsample): Sequential(
                (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): FrozenBatchNorm2d(256, eps=1e-05)
              )
            )
            (1): Bottleneck(
              (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(64, eps=1e-05)
              (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(64, eps=1e-05)
              (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(256, eps=1e-05)
              (relu): ReLU(inplace=True)
            )
            (2): Bottleneck(
              (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(64, eps=1e-05)
              (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(64, eps=1e-05)
              (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(256, eps=1e-05)
              (relu): ReLU(inplace=True)
            )
          )
          (layer2): Sequential(
            (0): Bottleneck(
              (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(128, eps=1e-05)
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(128, eps=1e-05)
              (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(512, eps=1e-05)
              (relu): ReLU(inplace=True)
              (downsample): Sequential(
                (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (1): FrozenBatchNorm2d(512, eps=1e-05)
              )
            )
            (1): Bottleneck(
              (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(128, eps=1e-05)
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(128, eps=1e-05)
              (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(512, eps=1e-05)
              (relu): ReLU(inplace=True)
            )
            (2): Bottleneck(
              (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(128, eps=1e-05)
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(128, eps=1e-05)
              (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(512, eps=1e-05)
              (relu): ReLU(inplace=True)
            )
            (3): Bottleneck(
              (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(128, eps=1e-05)
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(128, eps=1e-05)
              (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(512, eps=1e-05)
              (relu): ReLU(inplace=True)
            )
          )
          (layer3): Sequential(
            (0): Bottleneck(
              (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(256, eps=1e-05)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(256, eps=1e-05)
              (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(1024, eps=1e-05)
              (relu): ReLU(inplace=True)
              (downsample): Sequential(
                (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (1): FrozenBatchNorm2d(1024, eps=1e-05)
              )
            )
            (1): Bottleneck(
              (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(256, eps=1e-05)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(256, eps=1e-05)
              (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(1024, eps=1e-05)
              (relu): ReLU(inplace=True)
            )
            (2): Bottleneck(
              (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(256, eps=1e-05)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(256, eps=1e-05)
              (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(1024, eps=1e-05)
              (relu): ReLU(inplace=True)
            )
            (3): Bottleneck(
              (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(256, eps=1e-05)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(256, eps=1e-05)
              (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(1024, eps=1e-05)
              (relu): ReLU(inplace=True)
            )
            (4): Bottleneck(
              (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(256, eps=1e-05)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(256, eps=1e-05)
              (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(1024, eps=1e-05)
              (relu): ReLU(inplace=True)
            )
            (5): Bottleneck(
              (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(256, eps=1e-05)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(256, eps=1e-05)
              (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(1024, eps=1e-05)
              (relu): ReLU(inplace=True)
            )
          )
          (layer4): Sequential(
            (0): Bottleneck(
              (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(512, eps=1e-05)
              (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(512, eps=1e-05)
              (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(2048, eps=1e-05)
              (relu): ReLU(inplace=True)
              (downsample): Sequential(
                (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (1): FrozenBatchNorm2d(2048, eps=1e-05)
              )
            )
            (1): Bottleneck(
              (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(512, eps=1e-05)
              (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(512, eps=1e-05)
              (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(2048, eps=1e-05)
              (relu): ReLU(inplace=True)
            )
            (2): Bottleneck(
              (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(512, eps=1e-05)
              (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(512, eps=1e-05)
              (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(2048, eps=1e-05)
              (relu): ReLU(inplace=True)
            )
          )
        )
        (fpn): FeaturePyramidNetwork(
          (inner_blocks): ModuleList(
            (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
            (1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
            (2): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
            (3): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
          )
          (layer_blocks): ModuleList(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (extra_blocks): LastLevelMaxPool()
        )
      )
      (rpn): RegionProposalNetwork(
        (anchor_generator): AnchorGenerator()
        (head): RPNHead(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (cls_logits): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
          (bbox_pred): Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (roi_heads): RoIHeads(
        (box_roi_pool): MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=(7, 7), sampling_ratio=2)
        (box_head): TwoMLPHead(
          (fc6): Linear(in_features=12544, out_features=1024, bias=True)
          (fc7): Linear(in_features=1024, out_features=1024, bias=True)
        )
        (box_predictor): FastRCNNPredictor(
          (cls_score): Linear(in_features=1024, out_features=2, bias=True)
          (bbox_pred): Linear(in_features=1024, out_features=8, bias=True)
        )
      )
    )



The train_step function is integral to the model training workflow, handling data preparation, loss computation, and backpropagation within each batch iteration. It adjusts the learning rate per epoch and maintains a checkpointing mechanism by storing the model configuration, facilitating persistent training across sessions.


```python
def train_step(dataloader, device, model, optimizer,  lr_scheduler, epoch, output_path ):
        model.train()
        
        for img_ids, images, boxes, classes in dataloader:
            images = [image.to(device) for image in images]
            targets = []
            for b, c in zip(boxes, classes):
                b = torch.as_tensor(b, dtype=torch.float32).to(device) 
                c = torch.as_tensor(c, dtype=torch.int64).to(device)  
                targets.append({'boxes': b, 'labels': c})
    
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
       

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        lr_scheduler.step()
        torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}, output_path)
        return losses
```

The test_model function runs the model in evaluation mode, computes the IoU and score metrics for predictions above a defined threshold


```python
def calculate_iou(boxA, boxB):
    #Extract bounding boxes coordinates
    x0_A, y0_A, x1_A, y1_A = boxA
    x0_B, y0_B, x1_B, y1_B = boxB
    
    # Get the coordinates of the intersection rectangle
    x0_I = max(x0_A, x0_B)
    y0_I = max(y0_A, y0_B)
    x1_I = min(x1_A, x1_B)
    y1_I = min(y1_A, y1_B)
    #Calculate width and height of the intersection area.
    width_I = x1_I - x0_I 
    height_I = y1_I - y0_I
    width_I = torch.clamp(torch.as_tensor(width_I), min=0)
    height_I = torch.clamp(torch.as_tensor(height_I), min=0)
    # Calculate the intersection area:
    intersection = width_I * height_I
    # Calculate the union area:
    width_A, height_A = x1_A - x0_A, y1_A - y0_A
    width_B, height_B = x1_B - x0_B, y1_B - y0_B
    union = (width_A * height_A) + (width_B * height_B) - intersection
    # Calculate the IoU:
    IoU = intersection/union
    # for plotting purpose 
    boxI = torch.tensor([x0_I, y0_I, width_I,height_I])
    # Return the IoU and intersection box
    return IoU, boxI  
```


```python
def test_model(dataloader, model, device, iou_threshold=0.5):
    model.eval()
    with torch.no_grad():
        ious = []
        scores = []
        model = model.to(device)
        for img_ids, images, gt_boxes, gt_classes in dataloader:
            images = [image.to(device) for image in images]
            predictions = model(images)
            
            for i, prediction in enumerate(predictions):
                detections = []
                annotations = []
                
                pred_boxes = prediction['boxes'].cpu().numpy()
                pred_scores = prediction['scores'].cpu().numpy()
                pred_labels = prediction['labels'].cpu().numpy()

                gt_boxes_i = gt_boxes[i].cpu().numpy()
                gt_classes_i = gt_classes[i].cpu().numpy()

                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    if score >= iou_threshold:
                        detections.append([box, score, label])
                        scores.append(score)
                    
                for box, label in zip(gt_boxes_i, gt_classes_i):
                    annotations.append([box, label])

                for pr_box, gt_box in zip(pred_boxes, gt_boxes_i):
                    iou = calculate_iou(pr_box, gt_box)
                    ious.append(iou[0].item())
                   
           

        mean_score = statistics.mean(scores) 
        iou_mean = statistics.mean(ious)
    return  mean_score, iou_mean


```

Prepares the model for training on the computational device with SGD optimizer and a learning rate scheduler that reduces the learning rate at specified milestones.


```python
device = cfg["training"]["device"]
model.train() 
model.to(device)
num_epochs = cfg["training"]["num_epochs"]
lr= cfg["training"]["learning_rate"]
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg["training"]["LR_STEPS"], gamma=cfg["training"]["LR_DECAY"])
```

Establishes two Plotly figures to display the evolution of training loss and evaluation metrics, updating after each epoch.


```python
fig_loss = go.FigureWidget([
    go.Scatter(x=[], y=[], mode='lines+markers', name='Loss')
])

fig_metrics = go.FigureWidget([
    go.Scatter(x=[], y=[], mode='lines+markers', name='Score'),
    go.Scatter(x=[], y=[], mode='lines+markers', name='IoU Mean')
])

fig_loss.update_layout(
    title='Loss Over Epochs',
    xaxis_title='Epoch',
    yaxis_title='Loss Value',
    template='plotly_dark'
)

fig_metrics.update_layout(
    title='Score and IoU Mean Over Epochs',
    xaxis_title='Epoch',
    yaxis_title='Metric Values',
    template='plotly_dark'
)

display(fig_loss, fig_metrics)

for epoch in range(num_epochs):
    epoch_loss = train_step(dataloader, device, model, optimizer, lr_scheduler, epoch, output_path)
    print("epoch, epoch_loss: ", epoch, epoch_loss.item())
    score, iou_mean = test_model(dataloader, model, device, iou_threshold=0.5)
    lr_scheduler.step()


    with fig_loss.batch_update():
        fig_loss.data[0].x = fig_loss.data[0].x + (epoch + 1,)
        fig_loss.data[0].y = fig_loss.data[0].y + (epoch_loss.item(),)
        clear_output(wait=True)
        display(fig_loss, fig_metrics)
    
    with fig_metrics.batch_update():
        fig_metrics.data[0].x = fig_metrics.data[0].x + (epoch + 1,)
        fig_metrics.data[0].y = fig_metrics.data[0].y + (score,)
        fig_metrics.data[1].x = fig_metrics.data[1].x + (epoch + 1,)
        fig_metrics.data[1].y = fig_metrics.data[1].y + (iou_mean,)
        clear_output(wait=True)
        display(fig_loss, fig_metrics)

```


    FigureWidget({
        'data': [{'mode': 'lines+markers',
                  'name': 'Loss',
                  'type': 'scatter',
                  'uid': '2480171d-d3bf-47ef-97af-65c04a5b52a6',
                  'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
                  'y': [0.14609640836715698, 0.12525548040866852, 0.17360559105873108,
                        0.09792201220989227, 0.05968263000249863, 0.08141018450260162,
                        0.061159007251262665, 0.050571441650390625,
                        0.06298171728849411, 0.1863749474287033, 0.03796028345823288,
                        0.030268479138612747, 0.059277232736349106,
                        0.026875589042901993, 0.060437701642513275,
                        0.05784182250499725, 0.04988718405365944, 0.02773047611117363,
                        0.0648515447974205, 0.04606582596898079, 0.031076734885573387,
                        0.043988220393657684, 0.015952587127685547,
                        0.03473927453160286, 0.03875768184661865, 0.052626706659793854,
                        0.02063792198896408, 0.02163185551762581, 0.030225761234760284,
                        0.04813114553689957, 0.02735665999352932, 0.028718333691358566,
                        0.019747095182538033, 0.051428236067295074,
                        0.034027863293886185, 0.03087586909532547, 0.05698104202747345,
                        0.04557713121175766, 0.03648767247796059, 0.04182562232017517,
                        0.030266406014561653, 0.06318115442991257,
                        0.023911597207188606, 0.049163538962602615,
                        0.030562039464712143, 0.01899592950940132,
                        0.041151706129312515, 0.027751486748456955,
                        0.016403837129473686, 0.0928986519575119]}],
        'layout': {'template': '...',
                   'title': {'text': 'Loss Over Epochs'},
                   'xaxis': {'title': {'text': 'Epoch'}},
                   'yaxis': {'title': {'text': 'Loss Value'}}}
    })



    FigureWidget({
        'data': [{'mode': 'lines+markers',
                  'name': 'Score',
                  'type': 'scatter',
                  'uid': '76cb2b41-7146-437d-af20-5a082000f01e',
                  'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
                  'y': [0.646953821182251, 0.7802439332008362, 0.8001278042793274,
                        0.8153014183044434, 0.8554952144622803, 0.8404536843299866,
                        0.8728398680686951, 0.8602343797683716, 0.909328043460846,
                        0.8897066116333008, 0.9202113151550293, 0.9181588292121887,
                        0.9204890727996826, 0.9286413788795471, 0.9412693977355957,
                        0.9427483677864075, 0.9294086694717407, 0.9337748289108276,
                        0.9421432614326477, 0.9361980557441711, 0.9291803240776062,
                        0.92903071641922, 0.9411259889602661, 0.9512324929237366,
                        0.9427394866943359, 0.9531397819519043, 0.950230062007904,
                        0.9571880102157593, 0.9460996389389038, 0.9570150971412659,
                        0.9561529159545898, 0.9528381824493408, 0.9585581421852112,
                        0.9530490636825562, 0.9564716815948486, 0.9576847553253174,
                        0.9598023891448975, 0.9575732946395874, 0.959248423576355,
                        0.9588683247566223, 0.9565334320068359, 0.9569564461708069,
                        0.9538270831108093, 0.9577147960662842, 0.9565143585205078,
                        0.9602468013763428, 0.9603508710861206, 0.961237370967865,
                        0.9598677158355713]},
                 {'mode': 'lines+markers',
                  'name': 'IoU Mean',
                  'type': 'scatter',
                  'uid': 'cf8c28ca-f4e2-4a43-92ac-62166ebf0309',
                  'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
                  'y': [0.5940048067188446, 0.6453640478549282, 0.7223366848714455,
                        0.7446567839498139, 0.7899784533956117, 0.8034137272774947,
                        0.8252728836583302, 0.8196455090928327, 0.8533034922419958,
                        0.8521594494235708, 0.8771223894530126, 0.881782948206446,
                        0.8847192558555085, 0.8888381448850851, 0.8878440512866618,
                        0.8935836648689031, 0.8932832815637364, 0.8969412915015337,
                        0.8928452970904823, 0.8998373012714765, 0.897306907979864,
                        0.8999048550497556, 0.9064450777986144, 0.9076261877288254,
                        0.908958026126642, 0.9077101455262985, 0.914120592650038,
                        0.9137187314657077, 0.910026295349384, 0.9115034841144314,
                        0.9161062294795277, 0.9184765396594929, 0.9172221404026063,
                        0.920327562740101, 0.9214907680342992, 0.9223992382638897,
                        0.9212485486766523, 0.9194624319611765, 0.9225179705366221,
                        0.9219313811194391, 0.9231564401302305, 0.9235716333078483,
                        0.923955032515947, 0.9228144794409999, 0.9230022303540281,
                        0.9241528979831716, 0.9233200106658236, 0.9243066663232943,
                        0.9242552378571605]}],
        'layout': {'template': '...',
                   'title': {'text': 'Score and IoU Mean Over Epochs'},
                   'xaxis': {'title': {'text': 'Epoch'}},
                   'yaxis': {'title': {'text': 'Metric Values'}}}
    })

![png](/img5/newplot1.png)
![png](/img5/newplot2.png)

```python
output_path = cfg["data"]["output"]
images_path = cfg['data']['data_path'] + "test/"
image_files = [f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
```

Performing inference on selected images, running the model to detect drones, and visualizing the results.


```python
fig = make_subplots(rows=2, cols=2)
for index, im in enumerate(image_files[1:5]):
    image = Image.open(images_path+im)
    image = F.to_tensor(image)
    img = np.array(255*image.clone().permute(1,2,0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.ascontiguousarray(img, dtype=np.uint8)
    with torch.no_grad():
        prediction = model([image])
    for i, bx in enumerate(prediction[0]['boxes']):
        if prediction[0]['scores'][i] > 0.7:
            p1 = (int(bx[0]) , int(bx[1]) )
            p2 = (int(bx[2] ), int(bx[3] ))
            cv2.rectangle(img, p1[::-1], p2[::-1], (255,0,0), 3)
    fig.add_trace(
            go.Image(z=img),
            row=(index // 2) + 1, col=(index % 2) + 1
    )
fig.update_layout(
    height=800, width=800,
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    grid=dict(rows=2, columns=2, pattern='independent'),
)   
fig.show()     
```
![png](/img5/output_27_0.png)
  

