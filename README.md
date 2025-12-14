# Awesome-Computer-Vision-Paper

## Computer Vision Core Research Directions

### 📋 Overall Table of Contents

<details>
<summary><strong>Click to expand the full directory structure</strong> (major directions are bolded; click links to jump to sections)</summary>

- **[1. Anomaly Detection](#anomaly-detection)**
  - [1.1 Unsupervised Anomaly Detection](#1-unsupervised-anomaly-detection)
    - [1.1.1 Full-Spectrum Unsupervised Anomaly Detection](#11-full-spectrum-unsupervised-anomaly-detection)
  - [1.2 Few-Shot Anomaly Segmentation](#2-few-shot-anomaly-segmentation)
    - [1.2.1 Class-Generalizable Few-Shot Anomaly Segmentation](#21-class-generalizable-few-shot-anomaly-segmentation)
  - [1.3 Lightweight General Anomaly Detection](#3-lightweight-general-anomaly-detection)
    - [1.3.1 General Anomaly Detection (AnomalyNCD)](#31-general-anomaly-detection-anomalyncd)
- **[2. Anomaly Generation](#anomaly-generation)**
  - [2.1 Controllable Image Generation](#1-controllable-image-generation)
    - [2.1.1 CutPaste Method Generation](#11-cutpaste-method-generation)
    - [2.1.2 GAN Generation](#12-gan-generation)
    - [2.1.3 Diffusion Generation](#13-diffusion-generation)
      - [2.1.3.1 Text-based Generation](#131-text-based-generation)
      - [2.1.3.2 Image-based Generation](#132-image-based-generation)
      - [2.1.3.3 Multi-Modal Generation](#133-multi-modal-generation)
        - [2.1.3.3.1 Text-Image Multi-Modal](#1331-text-image-multi-modal)
        - [2.1.3.3.2 Image-Depth Multi-Modal](#1332-image-depth-multi-modal)
    - [2.1.4 Feature-level Anomaly Generation](#14-feature-level-anomaly-generation)
  - [2.2 Precise Mask](#2-precise-mask)
  - [2.3 Generation Quality Judgment and Evaluation System](#3-generation-quality-judgment-and-evaluation-system)
  - [2.4 Improving Generation Speed](#4-improving-generation-speed)
- **[3. Diffusion Model-Driven Image Fusion](#diffusion-model-driven-image-fusion)**
  - [3.1 Diffusion Transformer-based Fusion](#1-diffusion-transformer-based-fusion)
  - [3.2 Variational Autoencoder-free Latent Diffusion](#2-variational-autoencoder-free-latent-diffusion)
- **[4. Advanced Visual Reasoning & Learning](#advanced-visual-reasoning--learning)**
  - [4.1 Reinforced Visual Segmentation Reasoning](#1-reinforced-visual-segmentation-reasoning)
    - [4.1.1 Unified Reinforced Reasoning for Segmentation](#11-unified-reinforced-reasoning-for-segmentation)
  - [4.2 Self-Supervised Spatial Understanding](#2-self-supervised-spatial-understanding)
    - [4.2.1 Self-Supervised Reinforcement Learning for Spatial Understanding](#21-self-supervised-reinforcement-learning-for-spatial-understanding)
  - [4.3 Vision-Language Alignment](#3-vision-language-alignment)
    - [4.3.1 Vision-Language Alignment with Semantic Hierarchy](#31-vision-language-alignment-with-semantic-hierarchy)
- **[5. 3D Visual Modeling & Learning](#3d-visual-modeling--learning)**
  - [5.1 3D Scene Generation](#1-3d-scene-generation)
    - [5.1.1 Semantic Occupancy-based 3D Scene Generation](#11-semantic-occupancy-based-3d-scene-generation)
  - [5.2 Point Cloud Efficient Learning](#2-point-cloud-efficient-learning)
    - [5.2.1 Parameter-Efficient Fine-Tuning for Point Clouds](#21-parameter-efficient-fine-tuning-for-point-clouds)

</details>

## Anomaly Detection
Anomaly detection represents a core research direction in computer vision, addressing the challenge of identifying rare, abnormal patterns in real-world scenarios (e.g., industrial defects, medical anomalies). These techniques break through the bottleneck of scarce abnormal samples, providing reliable solutions for high-precision inspection tasks.

### Why This Research Direction Matters
- **Academic Value**: Advances unsupervised and few-shot learning paradigms, breaking through bottlenecks like "full-spectrum anomaly detection". Recent works (e.g., CVPR 2025) push the boundaries of anomaly pattern modeling and generalization.
- **Practical Production Value**: Reduces reliance on labeled anomaly data in industrial inspection and medical imaging, improving detection accuracy and enabling low-cost deployment.
- **Applicable Domains**: Industrial defect inspection, medical anomaly diagnosis, security surveillance.

### 📋 Hierarchical Subcategories
To clarify the structure, here's a numbered outline of the categories (major categories are **bolded**; subcategories are indented for easy navigation). Click links to jump to sections.
- **[1. Unsupervised Anomaly Detection](#1-unsupervised-anomaly-detection)**
  - [1.1 Full-Spectrum Unsupervised Anomaly Detection](#11-full-spectrum-unsupervised-anomaly-detection)
- **[2. Few-Shot Anomaly Segmentation](#2-few-shot-anomaly-segmentation)**
  - [2.1 Class-Generalizable Few-Shot Anomaly Segmentation](#21-class-generalizable-few-shot-anomaly-segmentation)
- **[3. Lightweight General Anomaly Detection](#3-lightweight-general-anomaly-detection)**
  - [3.1 General Anomaly Detection (AnomalyNCD)](#31-general-anomaly-detection-anomalyncd)

### 1. Unsupervised Anomaly Detection
Unsupervised anomaly detection focuses on identifying abnormal patterns without relying on labeled anomaly data, which is highly practical in real-world scenarios where anomalies are rare and difficult to label. This direction addresses the core pain point of traditional supervised methods that require large-scale labeled anomaly samples.

#### 1.1 Full-Spectrum Unsupervised Anomaly Detection
Full-spectrum unsupervised anomaly detection unifies detection frameworks for all types of anomalies (e.g., surface defects, structural deformations, functional anomalies) without labeled anomaly data. This eliminates the need for domain-specific tuning and reduces deployment costs in manufacturing.
- One Dinomaly2 Detect Them All: A Unified Framework for Full-Spectrum Unsupervised Anomaly Detection [[arXiv 2025]](https://arxiv.org/abs/2510.17611)[[code]](https://github.com/HUST-SLOW/AnomalyNCD)

### 2. Few-Shot Anomaly Segmentation
Few-shot anomaly segmentation aims to segment abnormal regions with only a small number of labeled anomaly samples, which is suitable for low-resource scenarios where labeled defect samples are scarce. This direction focuses on improving the generalization ability of models to unseen anomaly classes.

#### 2.1 Class-Generalizable Few-Shot Anomaly Segmentation
Class-generalizable few-shot anomaly segmentation enables segmentation of unseen anomaly classes with only a few samples, leveraging dictionary-based representation learning to capture universal anomaly patterns. This is vital for low-resource industrial scenarios where labeled defect samples are scarce.
- DictAS: A Framework for Class-Generalizable Few-Shot Anomaly Segmentation via Dictionary Lookup [[ICCV 2025]](https://arxiv.org/abs/2508.13560)[[code]](https://github.com/xiaozhen228/DictAS)

### 3. Lightweight General Anomaly Detection
Lightweight general anomaly detection focuses on developing efficient, low-complexity models that can be deployed on edge devices. This direction addresses the demand for high-throughput and low-cost anomaly detection in industrial inspection pipelines.

#### 3.1 General Anomaly Detection (AnomalyNCD)
AnomalyNCD (Nearest Class Distance) is a lightweight, general-purpose anomaly detection method that avoids complex generative models, focusing on distance-based outlier scoring for fast and scalable detection. It is suitable for edge devices and high-throughput inspection pipelines.
- AnomalyNCD: Towards Novel Anomaly Class Discovery in Industrial Scenarios [[CVPR 2025]](https://arxiv.org/abs/2410.14379) [[code]](https://github.com/HUST-SLOW/AnomalyNCD)

## Anomaly Generation
Anomaly Generation refers to the synthesis of artificial anomalous data to augment scarce real-world defect samples in industrial settings. This technique addresses the fundamental challenge of data imbalance in anomaly detection, where normal samples vastly outnumber anomalies, hindering model training.

### Why Anomaly Generation Matters
- **Academic Value**: It advances unsupervised and few/zero-shot learning paradigms by providing diverse, controllable synthetic data for benchmarking and evaluation. Recent surveys highlight its role in exploring generative models (e.g., diffusion-based) and vision-language integration, fostering innovations in evaluation frameworks and cross-domain generalization [[arXiv 2025]](https://arxiv.org/abs/2506.09368).
- **Practical Production Value**: In manufacturing pipelines, it reduces reliance on costly manual labeling, accelerates model training, and improves detection accuracy by simulating rare defects, minimizing downtime and quality control costs.
- **Applicable Domains**: Widely used in industrial inspection (e.g., semiconductors, automotive parts), medical imaging (e.g., tumor simulation), autonomous driving (e.g., road hazard synthesis), and surveillance (e.g., behavioral anomalies).

### 📋 Hierarchical Subcategories
To clarify the structure, here's a numbered outline of the categories (major categories are **bolded**; subcategories are indented for easy navigation). Click links to jump to sections.
- **[1. Controllable Image Generation](#1-controllable-image-generation)**
  - [1.1 CutPaste Method Generation](#11-cutpaste-method-generation)
  - [1.2 GAN Generation](#12-gan-generation)
  - [1.3 Diffusion Generation](#13-diffusion-generation)
    - [1.3.1 Text-based Generation](#131-text-based-generation)
    - [1.3.2 Image-based Generation](#132-image-based-generation)
    - [1.3.3 Multi-Modal Generation](#133-multi-modal-generation)
      - [1.3.3.1 Text-Image Multi-Modal](#1331-text-image-multi-modal)
      - [1.3.3.2 Image-Depth Multi-Modal](#1332-image-depth-multi-modal)
  - [1.4 Feature-level Anomaly Generation](#14-feature-level-anomaly-generation)
- **[2. Precise Mask](#2-precise-mask)**
- **[3. Generation Quality Judgment and Evaluation System](#3-generation-quality-judgment-and-evaluation-system)**
- **[4. Improving Generation Speed](#4-improving-generation-speed)**

### 1. Controllable Image Generation
Controllability allows precise specification of anomaly types, locations, and attributes via prompts or priors, ensuring synthetic data aligns with domain-specific needs. This boosts model generalization by simulating targeted scenarios, bridging the gap between generic augmentation and real defect variability.

#### 1.1 CutPaste Method Generation
CutPaste-inspired methods simulate anomalies through simple patch cutting and pasting from normal images, offering lightweight, label-free augmentation. This is vital for self-supervised anomaly detection, as it mimics realistic defects efficiently without requiring generative models, promoting accessibility in early-stage research and low-resource setups.
- CutPaste: Self-supervised Learning for Anomaly Detection and Localization [[ICCV 2021]](http://arxiv.org/pdf/2104.04015) [[unofficial code]](https://github.com/Runinho/pytorch-cutpaste)
- Natural Synthetic Anomalies for Self-supervised Anomaly Detection and Localization [[ECCV 2022]](https://arxiv.org/pdf/2109.15222.pdf) [[code]](https://github.com/hmsch/natural-synthetic-anomalies)

#### 1.2 GAN Generation
GAN-based approaches excel in producing high-fidelity, diverse anomalies by adversarially learning defect distributions from limited samples. Their importance lies in handling extreme class imbalance, enabling robust data augmentation for supervised fine-tuning and improving detection in domains like textiles where real defects are rare and varied.
- Multistage GAN for Fabric Defect Detection [[2019]](https://pubmed.ncbi.nlm.nih.gov/31870985/) [[code]](https://github.com/your-code-link-here)
- GAN-based Defect Synthesis for Anomaly Detection in Fabrics [[2020]](https://www.lfb.rwth-aachen.de/bibtexupload/pdf/RIP20c.pdf) [[code]](https://github.com/your-code-link-here)
- Defect Image Sample Generation with GAN for Improving Defect Recognition [[2020]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9000806) [[code]](https://github.com/your-code-link-here)
- Defective Samples Simulation through Neural Style Transfer for Automatic Surface Defect Segment [[2020]](http://arxiv.org/pdf/1910.03334) [[code]](https://github.com/your-code-link-here)
- Defect Transfer GAN: Diverse Defect Synthesis for Data Augmentation [[BMVC 2022]](https://openreview.net/pdf?id=2hMEdc35xZ6) [[code]](https://github.com/your-code-link-here)
- Defect-GAN: High-fidelity Defect Synthesis for Automated Defect Inspection [[2021]](https://dr.ntu.edu.sg/bitstream/10356/146285/2/WACV_2021_Defect_GAN__Camera_Ready_.pdf) [[code]](https://github.com/your-code-link-here)
- EID-GAN: Generative Adversarial Nets for Extremely Imbalanced Data Augmentation [[TII 2022]](https://ieeexplore.ieee.org/document/9795891) [[code]](https://github.com/your-code-link-here)

#### 1.3 Diffusion Generation
Diffusion models provide iterative denoising for superior sample quality and flexibility in anomaly synthesis. They are essential for modern controllable generation, allowing fine-grained control over defect attributes and enabling zero/few-shot adaptation, which drives advancements in scalable, high-resolution industrial simulations.

##### 1.3.1 Text-based Generation
Text-based generation harnesses natural language prompts to specify anomaly types, locations, and attributes, offering intuitive and flexible control for zero-shot synthesis. This approach excels in scenarios requiring semantic guidance without visual exemplars, fostering diverse and semantically coherent anomaly creation through prompt engineering and language model integration in diffusion processes.
- Component-aware Unsupervised Logical Anomaly Generation for Industrial Anomaly Detection [[2025]](https://arxiv.org/abs/2502.11712) [[code]](https://github.com/your-code-link-here)
- Photovoltaic Defect Image Generator with Boundary Alignment Smoothing Constraint for Domain Shift Mitigation [[2025]](https://arxiv.org/abs/2505.06117) [[code]](https://github.com/your-code-link-here)
- SeaS: Few-shot Industrial Anomaly Image Generation with Separation and Sharing Fine-tuning [[ICCV 2025]](https://arxiv.org/pdf/2410.14987) [[code]](https://github.com/HUST-SLOW/SeaS)
- Anomaly Anything: Promptable Unseen Visual Anomaly Generation [[CVPR 2025]](https://arxiv.org/abs/2406.01078) [[code]](https://github.com/EPFL-IMOS/AnomalyAny)
- AnoStyler: Text-Driven Localized Anomaly Generation via Lightweight Style Transfer [[AAAI 2026]](https://arxiv.org/abs/2511.06687) [[code]](https://github.com/yulimso/AnoStyler)

##### 1.3.2 Image-based Generation
Image-based generation conditions synthesis on visual cues like masks, bounding boxes, or reference images, enabling precise spatial localization and structural fidelity in anomaly placement. It is particularly effective for few-shot adaptation and boundary-aligned defects, enhancing realism in industrial simulations by leveraging existing visual priors to guide diffusion denoising.
- AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/28696) [[code]](https://github.com/sjtuplayer/anomalydiffusion)
- CAGEN: Controllable Anomaly Generator using Diffusion Model [[ICASSP 2024]](https://ieeexplore.ieee.org/document/10447663) [[code]](https://github.com/your-code-link-here)
- Few-Shot Anomaly-Driven Generation for Anomaly Classification and Segmentation [[ECCV 2024]](https://csgaobb.github.io/Pub_files/ECCV2024_AnoGen_CR_0730_Mobile.pdf) [[code]](https://github.com/gaobb/AnoGen)
- Bounding Box-Guided Diffusion for Synthesizing Industrial Images and Segmentation Map [[CVPRW 2025]](https://arxiv.org/abs/2505.03623) [[code]](https://github.com/covisionlab/diffusion_labeling)
- Enhancing Glass Defect Detection with Diffusion Models: Addressing Imbalanced Datasets in Manufacturing Quality Control [[2025]](https://arxiv.org/abs/2505.03134) [[code]](https://github.com/your-code-link-here)
- Anodapter: A Unified Framework for Generating Aligned Anomaly Images and Masks Using Diffusion Models [[2025]](https://ieeexplore.ieee.org/document/11000123) [[code]](https://github.com/your-code-link-here)

##### 1.3.3 Multi-Modal Generation
Multi-modal synthesis integrates data from diverse sources (e.g., RGB + depth + text), capturing richer contextual cues for robust detection in complex scenes. It's vital for handling incomplete or noisy inputs in real-world applications, improving cross-modal fusion and overall system resilience—especially within diffusion frameworks, where modalities can be jointly denoised for coherent anomaly injection.

###### 1.3.3.1 Text-Image Multi-Modal
Text-image multi-modal generation combines natural language descriptions with RGB visuals to guide anomaly synthesis, enabling semantically rich and contextually aware defect creation. This fusion enhances controllability and realism by leveraging textual semantics to refine visual outputs, ideal for scenarios blending descriptive prompts with image priors.
- AnomalyXFusion: Multi-modal Anomaly Synthesis with Diffusion [[2024]](https://arxiv.org/abs/2404.19444) [[data]](https://github.com/hujiecpp/MVTec-Caption) [[code]](https://github.com/your-code-link-here)
- A Novel Approach to Industrial Defect Generation through Blended Latent Diffusion Model with Online Adaptation [[2024]](https://arxiv.org/abs/2402.19330) [[code]](https://github.com/GrandpaXun242/AdaBLDM)
- AnomalyControl: Learning Cross-modal Semantic Features for Controllable Anomaly Synthesis [[2024]](https://arxiv.org/abs/2412.06510) [[code]](https://github.com/your-code-link-here)
- AnomalyControl: Highly-Aligned Anomalous Image Generation with Controlled Diffusion Model [[ACM MM 2025]](https://dl.acm.org/doi/abs/10.345/3746027.3755274) [[code]](https://github.com/your-code-link-here)
- Anomagic: Crossmodal Prompt-driven Zero-shot Anomaly Generation [[AAAI 2026]](https://arxiv.org/abs/2511.10020) [[code]](https://github.com/yuxin-jiang/Anomagic)

###### 1.3.3.2 Image-Depth Multi-Modal
Image-depth multi-modal generation fuses RGB images with depth maps to produce geometrically accurate anomalies, simulating 3D structural defects like deformations or occlusions. This approach is crucial for depth-sensitive industrial applications, ensuring spatial coherence and enhanced detection in 3D-aware environments through joint modality conditioning.
- AnomalyHybrid: A Domain-agnostic Generative Framework for General Anomaly Detection [[CVPR 2025 SyntaGen Workshop]](https://openaccess.thecvf.com/content/CVPR2025W/SyntaGen/papers/Zhao_AnomalyHybrid_A_Domain-agnostic_Generative_Framework_for_General_Anomaly_Detection_CVPRW_2025_paper.pdf) [[code]](https://github.com/your-code-link-here)

#### 1.4 Feature-level Anomaly Generation
Feature-level anomaly generation operates in latent or feature spaces to inject anomalies at higher abstractions, allowing for subtle and semantically meaningful defects without direct pixel-level manipulations. This method enhances efficiency, preserves global image consistency, and enables boundary-guided synthesis for more realistic industrial defect simulation.
- A Unified Anomaly Synthesis Strategy with Gradient Ascent for Few-shot Industrial Anomaly Detection [[ECCV 2024]](https://arxiv.org/abs/2407.09359) [[code]](https://github.com/Chern93/GLASS)
- Progressive Boundary Guided Anomaly Synthesis for Industrial Anomaly Detection [[TCSVT 2024]](https://ieeexplore.ieee.org/document/10716437) [[code]](https://github.com/cqylunlun/PBAS)

### 2. Precise Mask
Generating pixel-accurate masks ensures anomalies are spatially aligned with defects, facilitating supervised fine-tuning and precise localization. This is essential for pixel-level tasks like segmentation, reducing false positives and enhancing interpretability in downstream detection pipelines.
- Anodapter: A Unified Framework for Generating Aligned Anomaly Images and Masks Using Diffusion Models [[2025]](https://ieeexplore.ieee.org/document/11000123) [[code]](https://github.com/your-code-link-here)
- Bounding Box-Guided Diffusion for Synthesizing Industrial Images and Segmentation Map [[CVPRW 2025]](https://arxiv.org/abs/2505.03623) [[code]](https://github.com/covisionlab/diffusion_labeling)
- AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/28696) [[code]](https://github.com/sjtuplayer/anomalydiffusion)
- SeaS: Few-shot Industrial Anomaly Image Generation with Separation and Sharing Fine-tuning [[ICCV 2025]](https://arxiv.org/pdf/2410.14987) [[code]](https://github.com/HUST-SLOW/SeaS)
- Anomagic: Crossmodal Prompt-driven Zero-shot Anomaly Generation [[AAAI 2026]](https://arxiv.org/abs/2511.10020) [[code]](https://github.com/yuxin-jiang/Anomagic)

### 3. Generation Quality Judgment and Evaluation System
Robust evaluation metrics quantify synthetic data's fidelity, diversity, and utility, preventing domain shifts that degrade detection performance. This subcategory enables standardized benchmarking, guiding method selection and iterative improvements for trustworthy anomaly synthesis.
- ASBench: Image Anomalies Synthesis Benchmark for Anomaly Detection [[2025]](https://arxiv.org/abs/2510.07927) [[code]](https://github.com/your-code-link-here)

### 4. Improving Generation Speed
Enhancing generation speed is crucial for scaling anomaly synthesis to large datasets, enabling real-time augmentation during training and deployment in high-throughput industrial environments. This reduces computational overhead, making diffusion-based methods viable for resource-constrained settings without sacrificing diversity.

## Diffusion Model-Driven Image Fusion
Diffusion model-driven image fusion focuses on integrating multi-source visual data (e.g., RGB, infrared, multi-spectral) using advanced diffusion models, achieving unified semantic modeling and fine-grained controllable fusion. This direction addresses the limitations of traditional fusion methods such as semantic distortion and lack of controllability.

### Why This Research Direction Matters
- **Academic Value**: Advances unified fusion paradigms by combining diffusion models and transformers, breaking through the bottleneck of balancing fusion quality and semantic consistency.
- **Practical Production Value**: Improves fusion accuracy and real-time performance, enabling reliable multi-source data integration in remote sensing and medical imaging.
- **Applicable Domains**: Remote sensing image analysis, medical image fusion, autonomous driving multi-sensor data integration.

### 📋 Hierarchical Subcategories
To clarify the structure, here's a numbered outline of the categories (major categories are **bolded**; subcategories are indented for easy navigation). Click links to jump to sections.
- **[1. Diffusion Transformer-based Fusion](#1-diffusion-transformer-based-fusion)**
- **[2. Variational Autoencoder-free Latent Diffusion](#2-variational-autoencoder-free-latent-diffusion)**

### 1. Diffusion Transformer-based Fusion
Diffusion Transformer (DiT) combines the strengths of diffusion models (high-quality generation) and transformers (long-range semantic modeling), enabling unified semantic modeling and controllable fusion. This approach excels in preserving structural details and semantic consistency across fused images.
- Towards Unified Semantic and Controllable Image Fusion: A Diffusion Transformer Approach [[TPAMI 2025]](https://arxiv.org/abs/2512.07170) [[code]](https://github.com/Henry-Lee-real/DiTFuse)
- Text-DiFuse: An Interactive Multi-Modal Image Fusion Framework based on Text-modulated Diffusion Model (NeurIPS 2024) [Paper][Code]
- TextFusion: Unveiling the Power of Textual Semantics for Controllable Image Fusion (Information Fusion 2025) [Paper][Code]

### 2. Variational Autoencoder-free Latent Diffusion
Traditional Latent Diffusion Models (LDMs) rely on Variational Autoencoders (VAEs) for latent space mapping, which introduces reconstruction errors and computational overhead. VAE-free LDMs eliminate this dependency, reducing latency and preserving image fidelity—critical for real-time fusion and generation tasks.
- Latent Diffusion Model without Variational Autoencoder [[arXiv 2025]](https://arxiv.org/abs/2510.15301)

## Advanced Visual Reasoning & Learning
Advanced visual reasoning and learning enhance model capabilities in semantic understanding, spatial perception, and cross-modal alignment, bridging the gap between low-level feature extraction and high-level semantic reasoning. These techniques are foundational for high-quality visual detection and generation tasks.

### Why This Research Direction Matters
- **Academic Value**: Promotes cross-task mutual learning and unified reasoning frameworks, advancing the boundaries of self-supervised learning and vision-language alignment.
- **Practical Production Value**: Improves model generalization and adaptability in complex scenarios, supporting high-precision segmentation and spatial understanding in autonomous driving.
- **Applicable Domains**: Autonomous driving perception, general segmentation, cross-modal content retrieval.

### 📋 Hierarchical Subcategories
To clarify the structure, here's a numbered outline of the categories (major categories are **bolded**; subcategories are indented for easy navigation). Click links to jump to sections.
- **[1. Reinforced Visual Segmentation Reasoning](#1-reinforced-visual-segmentation-reasoning)**
  - [1.1 Unified Reinforced Reasoning for Segmentation](#11-unified-reinforced-reasoning-for-segmentation)
- **[2. Self-Supervised Spatial Understanding](#2-self-supervised-spatial-understanding)**
  - [2.1 Self-Supervised Reinforcement Learning for Spatial Understanding](#21-self-supervised-reinforcement-learning-for-spatial-understanding)
- **[3. Vision-Language Alignment](#3-vision-language-alignment)**
  - [3.1 Vision-Language Alignment with Semantic Hierarchy](#31-vision-language-alignment-with-semantic-hierarchy)

### 1. Reinforced Visual Segmentation Reasoning
Reinforced visual segmentation reasoning integrates reinforcement learning into segmentation tasks to optimize reasoning paths, enabling unified handling of multiple segmentation tasks and improving model generalization. This direction bridges the gap between low-level feature extraction and high-level semantic reasoning.

#### 1.1 Unified Reinforced Reasoning for Segmentation
Unified reinforced reasoning unifies segmentation tasks (e.g., object, scene, anomaly segmentation) into a single framework, leveraging reinforcement learning to optimize reasoning paths. This improves generalization across diverse segmentation scenarios and reduces task-specific tuning costs.
- LENS: Learning to Segment Anything with Unified Reinforced Reasoning [[AAAI 2026 Oral]](https://arxiv.org/abs/2508.14153)[[code]](https://github.com/hustvl/LENS)

### 2. Self-Supervised Spatial Understanding
Self-supervised spatial understanding aims to enhance models' ability to perceive spatial information (e.g., 3D geometry, scene layout) without labeled spatial data. This direction is critical for 3D scene generation, autonomous driving perception, and other tasks that require accurate spatial awareness.

#### 2.1 Self-Supervised Reinforcement Learning for Spatial Understanding
Self-supervised reinforcement learning (SSRL) enhances spatial understanding (e.g., 3D geometry, scene layout) without labeled spatial data. Spatial-SSRL focuses on spatial-aware reward design, enabling models to learn spatial priors from unlabeled data—critical for 3D scene generation and autonomous driving perception.
- Spatial-SSRL: Enhancing Spatial Understanding via Self-Supervised Reinforcement Learning [[arXiv 2025]](https://arxiv.org/abs/2510.27606)[[code]](https://github.com/InternLM/Spatial-SSRL)

### 3. Vision-Language Alignment
Vision-language alignment focuses on bridging the semantic gap between visual data and natural language, enabling more accurate cross-modal interaction. This direction is important for cross-modal retrieval, generation, and anomaly description in industrial scenarios.

#### 3.1 Vision-Language Alignment with Semantic Hierarchy
Vision-language alignment with semantic hierarchy and monotonicity addresses the "semantic gap" between images and text by modeling hierarchical semantic relationships (e.g., object → attribute → scene). This improves cross-modal retrieval, generation, and anomaly description in industrial scenarios.
- HiMo-CLIP: Modeling Semantic Hierarchy and Monotonicity in Vision-Language Alignment [[AAAI 2026 Oral ]](https://arxiv.org/abs/2511.06653)[[code]](github.com/UnicomAI/HiMo-CLIP)

## 3D Visual Modeling & Learning
3D visual modeling and learning focus on creating semantically consistent 3D scenes and efficiently learning point cloud representations, addressing challenges like cross-task inconsistency and high computational costs in 3D modeling. This direction is key for autonomous driving, AR/VR, and robotics.

### Why This Research Direction Matters
- **Academic Value**: Advances semantic occupancy-based 3D modeling and parameter-efficient learning paradigms, breaking through bottlenecks in low-resource point cloud learning.
- **Practical Production Value**: Reduces computational costs of 3D models and enables low-cost deployment on edge devices, supporting real-time 3D scene reconstruction in autonomous driving.
- **Applicable Domains**: Autonomous driving 3D perception, AR/VR scene generation, robotics environment modeling.

### 📋 Hierarchical Subcategories
To clarify the structure, here's a numbered outline of the categories (major categories are **bolded**; subcategories are indented for easy navigation). Click links to jump to sections.
- **[1. 3D Scene Generation](#1-3d-scene-generation)**
  - [1.1 Semantic Occupancy-based 3D Scene Generation](#11-semantic-occupancy-based-3d-scene-generation)
- **[2. Point Cloud Efficient Learning](#2-point-cloud-efficient-learning)**
  - [2.1 Parameter-Efficient Fine-Tuning for Point Clouds](#21-parameter-efficient-fine-tuning-for-point-clouds)

### 1. 3D Scene Generation
3D scene generation focuses on creating semantically consistent and realistic 3D scenes, which is a key technology for autonomous driving, AR/VR, and robotics. Semantic occupancy-based methods unify multiple 3D tasks, reducing redundancy in multi-task training.

#### 1.1 Semantic Occupancy-based 3D Scene Generation
Semantic occupancy-based cross-task mutual learning unifies multiple 3D tasks (e.g., reconstruction, segmentation, generation) by modeling semantic occupancy (voxel-level semantic presence). This reduces redundancy in multi-task training and improves 3D scene consistency.
- OccScene: Semantic Occupancy-based Cross-task Mutual Learning for 3D Scene Generation [[TPAMI 2025]](https://arxiv.org/abs/2412.11183)

### 2. Point Cloud Efficient Learning
Point cloud efficient learning focuses on reducing the computational cost and number of trainable parameters for point cloud models, enabling low-cost adaptation to new domains/tasks. This is critical for edge deployment of point cloud-based applications.

#### 2.1 Parameter-Efficient Fine-Tuning for Point Clouds
Parameter-efficient fine-tuning (PEFT) in the spectral domain reduces the number of trainable parameters for point cloud models, enabling low-cost adaptation to new domains/tasks without full retraining. This is critical for edge deployment of point cloud-based anomaly detection/3D reconstruction.
- Parameter-Efficient Fine-Tuning in Spectral Domain for Point Cloud Learning [[TPAMI 2025]](https://arxiv.org/abs/2410.08114)[[code]](https://github.com/jerryfeng2003/PointGST)

💌 **Acknowledgement**  
We acknowledge the open-source community (e.g., CVPR 2025, AAAI 2026, ArXiv) for providing access to cutting-edge research papers and code repositories. Big thanks to the authors of these works for advancing the field of computer vision! We also acknowledge the [Awesome Industrial Anomaly Detection](https://github.com/M-3LAB/awesome-industrial-anomaly-detection) repository for its comprehensive paper list and datasets on industrial image anomaly/defect detection. Big thanks to this amazing open-source work!
