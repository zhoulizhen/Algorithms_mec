# Chasing Common Knowledge: Joint Large Model Selection and Pulling in MEC with Parameter Sharing

## Overview

This repository provides the code implementation for our paper, where we tackle the joint large model selection and pulling problem in Mobile Edge Computing (MEC) networks. As Pretrained Foundation Models (PFMs) are increasingly fine-tuned to meet personalized inference demands, the computational load on remote data centers becomes overwhelming. MEC offers a solution by enabling fine-tuned PFMs to be deployed closer to users at cloudlets.

The key challenge addressed in this work is the resource-intensive, delay-sensitive and cost-prohibitive nature of executing large models at the edge. Our approach explores parameter sharing among fine-tuned models based on their common knowledge, and we develop novel algorithms to reduce total delay, operating cost, and improve inference accuracy. 

### Key Contributions:
- Formulation of a **Non-Linear Integer Programming (NLIP)** problem to minimize the total delay of inference requests.
- Transformation of the NLIP into a simpler **Integer Linear Program (ILP)**.
- Development of a **randomized algorithm** with a provable approximation ratio.
- Design of an **online learning algorithm** with bounded regret using the multi-armed bandit technique for dynamic admissions of requests.
- **Extensive experiments** demonstrating a 38% reduction in total delays and costs, along with an 5% improvement in inference accuracy.

## Installation

You can install the required dependencies using pip:

```bash

pip install -r requirements.txt
