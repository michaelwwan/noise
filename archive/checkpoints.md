## Archive: Model Checkpoints

*3/25/26 Note:* This is an archive of the checkpoints provided with the release of our original [**CVPR CVMI 2024**] paper. Back then, we split up our original mouse (M1–M5) and human (H1–H2) datasets into training and test splits to enable cross-validation, but this led to an complicated set of correspondingly pretrained models, documented here. In our new [**TBD 2026**] paper, we trained new models on the entire mouse and human datasets, yielding the updated models presented in the [main readme](/README.md).


[Here](https://drive.google.com/drive/folders/1K7MrAER_c5vpgw2TRb3pqrMeZSqn4MrB?usp=share_link) is a link to the checkpoint files.

You can follow our recommendations below for which model to use, or take a look at our [paper](https://arxiv.org/pdf/2404.10130.pdf) for more information about each model.

| Experiment | Model Filename | Description | Use This Model For ... |
| ----- | ----- | ----- | ----- |
| YOLOv8 M→M | `yolo_m1234.pt` | YOLOv8 model trained on M1, M2, M3, M4 | **Mouse osteoclast** instance segmentation (use this, or any of the other four YOLOv8 M→M models below) |
| YOLOv8 M→M | `yolo_m1235.pt` | YOLOv8 model trained on M1, M2, M3, M5 | |
| YOLOv8 M→M | `yolo_m1245.pt` | YOLOv8 model trained on M1, M2, M4, M5 | |
| YOLOv8 M→M | `yolo_m1345.pt` | YOLOv8 model trained on M1, M3, M4, M5 | |
| YOLOv8 M→M | `yolo_m2345.pt` | YOLOv8 model trained on M2, M3, M4, M5 | | 
| NOISe M→H and H→H | `yolo_m_det_only.pt` | YOLOv8 model trained on mouse osteoclast & nuclei bounding boxes for detection only, used as multiclass pretraining step in the NOISe models  | **Developing a NOISe model** in a new domain (by fine-tuning this on instance segmentation masks in that domain) |
| NOISe M→H | `noise_m.pt` | NOISe model trained on M1–5 (i.e., `yolo_m_det_only.pt`, fine-tuned on M1–5 instance segmentation masks) | **Mouse, human, or new domain osteoclast** instance segmentation (try this alongside the other recommended models) |
| NOISe H→H | `noise_m_h1.pt` | NOISe model trained on H1 (i.e., `yolo_m_det_only.pt`, fine-tuned on H1 instance segmentation masks) | **Human osteoclast** instance segmentation (use this, or the NOISe H→H model below)  |
| NOISe H→H | `noise_m_h1.pt` | NOISe model trained on H2 (i.e., `yolo_m_det_only.pt`, fine-tuned on H2 instance segmentation masks) |
