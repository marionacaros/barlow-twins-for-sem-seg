# Self Supervised Barlow Twins for Semantic Segmentation of aerial LiDAR data

Created by [Mariona Carós](https://www.linkedin.com/in/marionacaros/), [Santi Seguí](https://ssegui.github.io/) and [Jordi Vitrià](https://algorismes.github.io/) from University of Barcelona and [Ariadna Just](https://www.linkedin.com/in/ariadna-just-0a667559/?originalSubdomain=es) from [Cartographic Institute of Catalonia](https://www.icgc.cat/es)

## Introduction
Airborne LiDAR systems have the capability to capture the Earth's surface  by generating extensive point cloud data comprised of points mainly defined by 3D coordinates. However, labeling such points for supervised learning tasks is time-consuming. As a result, there is a need to investigate techniques that can learn from unlabeled data in order to significantly reduce the number of annotated samples. In this work, we propose to train a self-supervised encoder with Barlow Twins and use it as a pre-trained network in the task of semantic scene segmentation. The experimental results demonstrate that our unsupervised pre-training boosts performance once fine-tuned on the supervised task, especially for under-represented categories.
