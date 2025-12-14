# Awesome-Computer-Vision-Paper

Computer Vision
Anomaly Detection & Generation

Anomaly detection and generation represent a core research direction in computer vision, addressing the challenge of identifying and synthesizing rare, abnormal patterns in real-world scenarios (e.g., industrial defects, medical anomalies). These techniques break through the bottleneck of scarce abnormal samples, providing reliable solutions for high-precision inspection tasks.

Why This Research Direction Matters

- Academic Value: Advances unsupervised and few-shot learning paradigms, breaking through bottlenecks like "full-spectrum anomaly detection". Recent works (e.g., CVPR 2025) push the boundaries of anomaly pattern modeling and generalization.

- Practical Production Value: Reduces reliance on labeled anomaly data in industrial inspection and medical imaging, improving detection accuracy and enabling low-cost deployment.

- Applicable Domains: Industrial defect inspection, medical anomaly diagnosis, security surveillance.

📋 Hierarchical Subcategories

To clarify the structure, here's a numbered outline of the categories (major categories are bolded; subcategories are indented for easy navigation). Click links to jump to sections.

- [1. Unsupervised Anomaly Detection](#unsupervised-anomaly-detection)[1.1 Full-Spectrum Unsupervised Anomaly Detection](#full-spectrum-unsupervised-anomaly-detection)

[2. Few-Shot Anomaly Segmentation](#few-shot-anomaly-segmentation)[2.1 Class-Generalizable Few-Shot Anomaly Segmentation](#class-generalizable-few-shot-anomaly-segmentation)

[3. Lightweight General Anomaly Detection](#lightweight-general-anomaly-detection)[3.1 General Anomaly Detection (AnomalyNCD)](#general-anomaly-detection-anomalyncd)


---
Diffusion Model-Driven Image Fusion

Diffusion model-driven image fusion focuses on integrating multi-source visual data (e.g., RGB, infrared, multi-spectral) using advanced diffusion models, achieving unified semantic modeling and fine-grained controllable fusion. This direction addresses the limitations of traditional fusion methods such as semantic distortion and lack of controllability.

Why This Research Direction Matters

- Academic Value: Advances unified fusion paradigms by combining diffusion models and transformers, breaking through the bottleneck of balancing fusion quality and semantic consistency.

- Practical Production Value: Improves fusion accuracy and real-time performance, enabling reliable multi-source data integration in remote sensing and medical imaging.

- Applicable Domains: Remote sensing image analysis, medical image fusion, autonomous driving multi-sensor data integration.

📋 Hierarchical Subcategories

To clarify the structure, here's a numbered outline of the categories (major categories are bolded; subcategories are indented for easy navigation). Click links to jump to sections.

- [1. Diffusion Transformer-based Fusion](#diffusion-transformer-based-fusion)

- [2. Variational Autoencoder-free Latent Diffusion](#vae-free-latent-diffusion)


---
Advanced Visual Reasoning & Learning

Advanced visual reasoning and learning enhance model capabilities in semantic understanding, spatial perception, and cross-modal alignment, bridging the gap between low-level feature extraction and high-level semantic reasoning. These techniques are foundational for high-quality visual detection and generation tasks.

Why This Research Direction Matters

- Academic Value: Promotes cross-task mutual learning and unified reasoning frameworks, advancing the boundaries of self-supervised learning and vision-language alignment.

- Practical Production Value: Improves model generalization and adaptability in complex scenarios, supporting high-precision segmentation and spatial understanding in autonomous driving.

- Applicable Domains: Autonomous driving perception, general segmentation, cross-modal content retrieval.

📋 Hierarchical Subcategories

To clarify the structure, here's a numbered outline of the categories (major categories are bolded; subcategories are indented for easy navigation). Click links to jump to sections.

- [1. Reinforced Visual Segmentation Reasoning](#reinforced-visual-segmentation-reasoning)[1.1 Unified Reinforced Reasoning for Segmentation](#unified-reinforced-reasoning-for-segmentation)

[2. Self-Supervised Spatial Understanding](#self-supervised-spatial-understanding)[2.1 Self-Supervised Reinforcement Learning for Spatial Understanding](#self-supervised-rl-for-spatial-understanding)

[3. Vision-Language Alignment](#vision-language-alignment)[3.1 Vision-Language Alignment with Semantic Hierarchy](#vision-language-alignment-with-semantic-hierarchy)


---
3D Visual Modeling & Learning

3D visual modeling and learning focus on creating semantically consistent 3D scenes and efficiently learning point cloud representations, addressing challenges like cross-task inconsistency and high computational costs in 3D modeling. This direction is key for autonomous driving, AR/VR, and robotics.

Why This Research Direction Matters

- Academic Value: Advances semantic occupancy-based 3D modeling and parameter-efficient learning paradigms, breaking through bottlenecks in low-resource point cloud learning.

- Practical Production Value: Reduces computational costs of 3D models and enables low-cost deployment on edge devices, supporting real-time 3D scene reconstruction in autonomous driving.

- Applicable Domains: Autonomous driving 3D perception, AR/VR scene generation, robotics environment modeling.

📋 Hierarchical Subcategories

To clarify the structure, here's a numbered outline of the categories (major categories are bolded; subcategories are indented for easy navigation). Click links to jump to sections.

- [1. 3D Scene Generation](#3d-scene-generation)[1.1 Semantic Occupancy-based 3D Scene Generation](#semantic-occupancy-3d-scene-generation)

[2. Point Cloud Efficient Learning](#point-cloud-efficient-learning)[2.1 Parameter-Efficient Fine-Tuning for Point Clouds](#parameter-efficient-fine-tuning-point-clouds)


---
1. Diffusion Transformer-based Fusion

Diffusion Transformer (DiT) combines the strengths of diffusion models (high-quality generation) and transformers (long-range semantic modeling), enabling unified semantic modeling and controllable fusion. This approach excels in preserving structural details and semantic consistency across fused images.

- Towards Unified Semantic and Controllable Image Fusion: A Diffusion Transformer Approach [[Preprint]](https://arxiv.org/search/cs?query=Towards+Unified+Semantic+and+Controllable+Image+Fusion%3A+A+Diffusion+Transformer+Approach&searchtype=title)

2. Variational Autoencoder-free Latent Diffusion

Traditional Latent Diffusion Models (LDMs) rely on Variational Autoencoders (VAEs) for latent space mapping, which introduces reconstruction errors and computational overhead. VAE-free LDMs eliminate this dependency, reducing latency and preserving image fidelity—critical for real-time fusion and generation tasks.

- Latent Diffusion Model without Variational Autoencoder [[Preprint]](https://arxiv.org/search/cs?query=Latent+Diffusion+Model+without+Variational+Autoencoder&searchtype=title)

1. Unsupervised Anomaly Detection

Unsupervised anomaly detection focuses on identifying abnormal patterns without relying on labeled anomaly data, which is highly practical in real-world scenarios where anomalies are rare and difficult to label. This direction addresses the core pain point of traditional supervised methods that require large-scale labeled anomaly samples.

1.1 Full-Spectrum Unsupervised Anomaly Detection

Full-spectrum unsupervised anomaly detection unifies detection frameworks for all types of anomalies (e.g., surface defects, structural deformations, functional anomalies) without labeled anomaly data. This eliminates the need for domain-specific tuning and reduces deployment costs in manufacturing.

- One Dinomaly2 Detect Them All: A Unified Framework for Full-Spectrum Unsupervised Anomaly Detection [[Preprint]](https://arxiv.org/search/cs?query=One+Dinomaly2+Detect+Them+All%3A+A+Unified+Framework+for+Full-Spectrum+Unsupervised+Anomaly+Detection&searchtype=title)

2. Few-Shot Anomaly Segmentation

Few-shot anomaly segmentation aims to segment abnormal regions with only a small number of labeled anomaly samples, which is suitable for low-resource scenarios where labeled defect samples are scarce. This direction focuses on improving the generalization ability of models to unseen anomaly classes.

2.1 Class-Generalizable Few-Shot Anomaly Segmentation

Class-generalizable few-shot anomaly segmentation enables segmentation of unseen anomaly classes with only a few samples, leveraging dictionary-based representation learning to capture universal anomaly patterns. This is vital for low-resource industrial scenarios where labeled defect samples are scarce.

- DictAS: A Framework for Class-Generalizable Few-Shot Anomaly Segmentation via Dictionary [[Preprint]](https://arxiv.org/search/cs?query=DictAS%3A+A+Framework+for+Class-Generalizable+Few-Shot+Anomaly+Segmentation+via+Dictionary&searchtype=title)

3. Lightweight General Anomaly Detection

Lightweight general anomaly detection focuses on developing efficient, low-complexity models that can be deployed on edge devices. This direction addresses the demand for high-throughput and low-cost anomaly detection in industrial inspection pipelines.

3.1 General Anomaly Detection (AnomalyNCD)

AnomalyNCD (Nearest Class Distance) is a lightweight, general-purpose anomaly detection method that avoids complex generative models, focusing on distance-based outlier scoring for fast and scalable detection. It is suitable for edge devices and high-throughput inspection pipelines.

- AnomalyNCD [[CVPR 2025]](https://arxiv.org/pdf/2504.04158) [[Project Page]](https://cvpr2025-jarvisir.github.io/) [[Github]](https://github.com/LYL1015/JarvisIR)

1. Reinforced Visual Segmentation Reasoning

Reinforced visual segmentation reasoning integrates reinforcement learning into segmentation tasks to optimize reasoning paths, enabling unified handling of multiple segmentation tasks and improving model generalization. This direction bridges the gap between low-level feature extraction and high-level semantic reasoning.

1.1 Unified Reinforced Reasoning for Segmentation

Unified reinforced reasoning unifies segmentation tasks (e.g., object, scene, anomaly segmentation) into a single framework, leveraging reinforcement learning to optimize reasoning paths. This improves generalization across diverse segmentation scenarios and reduces task-specific tuning costs.

- LENS: Learning to Segment Anything with Unified Reinforced Reasoning [[Preprint]](https://arxiv.org/search/cs?query=LENS%3A+Learning+to+Segment Anything+with+Unified+Reinforced+Reasoning&searchtype=title)

2. Self-Supervised Spatial Understanding

Self-supervised spatial understanding aims to enhance models' ability to perceive spatial information (e.g., 3D geometry, scene layout) without labeled spatial data. This direction is critical for 3D scene generation, autonomous driving perception, and other tasks that require accurate spatial awareness.

2.1 Self-Supervised Reinforcement Learning for Spatial Understanding

Self-supervised reinforcement learning (SSRL) enhances spatial understanding (e.g., 3D geometry, scene layout) without labeled spatial data. Spatial-SSRL focuses on spatial-aware reward design, enabling models to learn spatial priors from unlabeled data—critical for 3D scene generation and autonomous driving perception.

- Spatial-SSRL: Enhancing Spatial Understanding via Self-Supervised Reinforcement Learning [[Preprint]](https://arxiv.org/search/cs?query=Spatial-SSRL%3A+Enhancing+Spatial+Understanding+via+Self-Supervised+Reinforcement+Learning&searchtype=title)

3. Vision-Language Alignment

Vision-language alignment focuses on bridging the semantic gap between visual data and natural language, enabling more accurate cross-modal interaction. This direction is important for cross-modal retrieval, generation, and anomaly description in industrial scenarios.

3.1 Vision-Language Alignment with Semantic Hierarchy

Vision-language alignment with semantic hierarchy and monotonicity addresses the "semantic gap" between images and text by modeling hierarchical semantic relationships (e.g., object → attribute → scene). This improves cross-modal retrieval, generation, and anomaly description in industrial scenarios.

- HiMo-CLIP: Modeling Semantic Hierarchy and Monotonicity in Vision-Language Alignment [[AAAI 2026 Oral]](https://arxiv.org/search/cs?query=HiMo-CLIP%3A+Modeling+Semantic+Hierarchy+and+Monotonicity+in+Vision-Language+Alignment&searchtype=title)

1. 3D Scene Generation

3D scene generation focuses on creating semantically consistent and realistic 3D scenes, which is a key technology for autonomous driving, AR/VR, and robotics. Semantic occupancy-based methods unify multiple 3D tasks, reducing redundancy in multi-task training.

1.1 Semantic Occupancy-based 3D Scene Generation

Semantic occupancy-based cross-task mutual learning unifies multiple 3D tasks (e.g., reconstruction, segmentation, generation) by modeling semantic occupancy (voxel-level semantic presence). This reduces redundancy in multi-task training and improves 3D scene consistency.

- OccScene: Semantic Occupancy-based Cross-task Mutual Learning for 3D Scene Generation [[Preprint]](https://arxiv.org/search/cs?query=OccScene%3A+Semantic+Occupancy-based+Cross-task+Mutual+Learning+for+3D+Scene+Generation&searchtype=title)

2. Point Cloud Efficient Learning

Point cloud efficient learning focuses on reducing the computational cost and number of trainable parameters for point cloud models, enabling low-cost adaptation to new domains/tasks. This is critical for edge deployment of point cloud-based applications.

2.1 Parameter-Efficient Fine-Tuning for Point Clouds

Parameter-efficient fine-tuning (PEFT) in the spectral domain reduces the number of trainable parameters for point cloud models, enabling low-cost adaptation to new domains/tasks without full retraining. This is critical for edge deployment of point cloud-based anomaly detection/3D reconstruction.

- Parameter-Efficient Fine-Tuning in Spectral Domain for Point Cloud Learning [[Preprint]](https://arxiv.org/search/cs?query=Parameter-Efficient+Fine-Tuning+in+Spectral+Domain+for+Point+Cloud+Learning&searchtype=title)


---
💌 Acknowledgement

We acknowledge the open-source community (e.g., CVPR 2025, AAAI 2026, ArXiv) for providing access to cutting-edge research papers and code repositories. Big thanks to the authors of these works for advancing the field of computer vision!
