# Spiking Neural Network for Patient vs Control Classification

## Table of Contents
- [Spiking Neural Network for Patient vs Control Classification](#spiking-neural-network-for-patient-vs-control-classification)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Background and Theory](#background-and-theory)
    - [Spiking Neural Networks (SNNs)](#spiking-neural-networks-snns)
    - [Leaky Integrate-and-Fire Neurons](#leaky-integrate-and-fire-neurons)
  - [Mathematical Model](#mathematical-model)
  - [Network Architecture](#network-architecture)
  - [Implementation Details](#implementation-details)
  - [Training Procedure](#training-procedure)
  - [Testing and Results](#testing-and-results)
  - [Accuracy and Performance](#accuracy-and-performance)
  - [Contributions](#contributions)
  - [Future Work](#future-work)
  - [How to Run](#how-to-run)
  - [References](#references)

---

## Project Overview

This project implements a biologically plausible spiking neural network (SNN) based on Leaky Integrate-and-Fire (LIF) neurons to classify subjects as **patients** or **healthy controls** using time series marker displacement data. The network uses rate-coded spike trains derived from continuous input features and learns via a supervised Hebbian-like learning rule with winner-takes-all competition.

The objective is to investigate the use of simple spiking neurons in clinical data classification, exploring biologically inspired mechanisms for pattern recognition.

---

## Background and Theory

### Spiking Neural Networks (SNNs)

SNNs are the third generation of neural network models that more closely resemble biological neural systems by processing and transmitting information as discrete spike events over time, rather than continuous activations. They incorporate temporal dynamics and event-driven computation, offering potential advantages in energy efficiency and temporal pattern recognition.

### Leaky Integrate-and-Fire Neurons

LIF neurons model the membrane potential \( v(t) \) which integrates incoming current, leaks over time, and generates a spike when a threshold \( v_{th} \) is reached. The membrane potential then resets. This dynamic is mathematically represented by:

\[
\tau \frac{dv}{dt} = -v(t) + I(t)
\]

where \( \tau \) is the membrane time constant and \( I(t) \) is the input current.

---

## Mathematical Model

1. **Membrane potential update:**

\[
v_{t+1} = v_t + \frac{\Delta t}{\tau}(-v_t + I_t)
\]

where \( \Delta t \) is the time step, and \( I_t \) is computed as weighted input spikes:

\[
I_t = \sum_{i=1}^{N} w_i s_i(t)
\]

with \( w_i \) as synaptic weights and \( s_i(t) \) as binary input spikes at time \( t \).

2. **Spike generation:**

If \( v_{t+1} \geq v_{th} \), neuron fires a spike and \( v_{t+1} \) resets to \( v_{reset} \).

3. **Learning rule:**

Weights \( w \) of the winner neuron update using Hebbian-like supervised learning:

\[
w^{new} = w^{old} + \eta (\bar{s}_{input} - w^{old})
\]

where \( \eta \) is learning rate, and \( \bar{s}_{input} \) is average input spike rate over the time window.

---

## Network Architecture

- **Input Layer:** Rate-coded spike trains generated from continuous feature vectors.
- **Output Layer:** Two LIF neurons representing two classes (patients, controls).
- **Weights:** Synaptic strengths initialized randomly and adapted during supervised training.
- **Competition:** Winner-takes-all based on total output spikes over a spike train window.

---

## Implementation Details

- **Programming Language:** Python 3  
- **Libraries:** NumPy for numerical operations  
- **Input encoding:** Rate coding with normalized spike probability per input feature  
- **Simulation:** Discrete time steps integrating input spikes into membrane potential  
- **Training:** Supervised Hebbian update on winner neuron weights  
- **Normalization:** Input vectors normalized per sample to [0,1]

---

## Training Procedure

- Each training vector is converted to a spike train (binary spikes across time steps).
- For each spike train, membrane potentials evolve, spikes are generated.
- The neuron with maximum spikes is the winner.
- If the winner matches the true class label, weights are updated to better represent that class.
- Training is repeated over multiple epochs for convergence.

---

## Testing and Results

- New unseen subjects’ feature vectors are converted to spike trains.
- Network run forward to generate output spikes.
- Classification assigned by neuron with maximum total spikes.
- Results reported for each test subject with predicted class (control or patient).

---

## Accuracy and Performance

- The model achieved an average classification accuracy of **[INSERT YOUR ACCURACY]%** on the test dataset.
- Training converged within **[INSERT NUMBER]** epochs.
- Weight adaptation showed stable convergence with bounded synaptic weights.
- Confusion matrix and detailed performance metrics can be added here if available.

---

## Contributions

- Developed a biologically inspired SNN using LIF neurons implemented from scratch.
- Designed rate coding for input spike train generation from continuous clinical data.
- Implemented supervised Hebbian learning with winner-takes-all competition.
- Validated model on patient and healthy control datasets.
- Provided detailed documentation, testing, and interpretation of results.

---

## Future Work

- Extend the network with more neurons and layers for richer representations.
- Implement temporal coding schemes beyond rate coding (e.g., latency coding).
- Explore unsupervised STDP learning rules for more autonomous feature extraction.
- Integrate with frameworks like Brian2 or BindsNET for advanced SNN simulation.
- Compare performance against classical machine learning models on same dataset.

---

## How to Run

1. Place your training data files `control.txt` and `patient.txt` in the project directory.  
2. Place test data file (e.g., `test_subjects.txt`) in the same directory.  
3. Run `python lif_spiking_network.py` (or your script filename).  
4. View classification results printed in the console.  
5. Modify hyperparameters in the script for experimentation.

---

## References

- Gerstner, W., & Kistler, W. M. (2002). *Spiking Neuron Models: Single Neurons, Populations, Plasticity.* Cambridge University Press.  
- Maass, W. (1997). Networks of spiking neurons: The third generation of neural network models. *Neural Networks.*  
- Izhikevich, E. M. (2004). Which model to use for cortical spiking neurons? *IEEE Transactions on Neural Networks.*
