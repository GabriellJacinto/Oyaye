# **NP-SNN Architecture (Fase 1) – Detalhamento Técnico Completo**

## **Visão Geral**

O NP-SNN (Neural Physics-Informed Spiking Neural Network) é um modelo projetado para aprender **trajetórias orbitais contínuas** e **respeitar as leis físicas** do movimento orbital enquanto recebe observações esparsas (RA/Dec, range, Doppler). Ele combina:

* **SNN (Spiking Neural Network)** para representação dinâmica contínua;
* **PINN (Physics-Informed Neural Network)** adicionando equações diferenciais orbitais como restrições;
* **Time Encoding** para transformar o tempo em um espaço de características rico;
* **Observation Decoder** para converter o estado latente em posição/velocidade observável;
* **Physics Losses** baseadas em J2, arrasto, SRP.

O objetivo é aprender: 
$$\hat{x}(t) = [\hat{\mathbf{r}}(t), \hat{\mathbf{v}}(t)] \in \mathbb{R}^6$$

onde:

* $\mathbf{r}(t)$ = posição (
  km)
* $\mathbf{v}(t)$ = velocidade (
  km/s)

## **Time Encoding**

O tempo contínuo é transformado em uma representação rica:

### **Fourier Features**

Para frequências $\omega_k$:

$$\gamma(t) = [\sin(2 \pi \omega_k t), \cos(2 \pi \omega_k t)]_{k=1}^K$$

Escolha típica: $K = 16$ ou $32$.

### **Encoding via MLP (aprendido)**

Um MLP aprende:

$$\mathbf{z}(t) = \text{MLP}(t)$$

Isso é útil quando escalas temporais variam entre órbitas. O encoding final é a concatenação:

$$\mathbf{e}(t) = [t, \gamma(t), \mathbf{z}(t)]$$

## **Núcleo SNN – Dinâmica espiking contínua**

O coração do modelo é um bloco de neurônios LIF ou RLIF.

### **Dinâmica do neurônio LIF**

$$\tau_m \frac{du}{dt} = -u + I(t)$$

O neurônio dispara quando $u \geq V_{th}$ e o potencial é resetado:

$$\text{spike}(t) = H(u(t) - V_{th})$$

com reset:

$$u(t^+) = u(t) - V_{th}$$

### **RLIF (LIF com decaimento adaptativo)**

Usamos RLIF para aprender temporalidades mais longas:

$$\tau_m \frac{du}{dt} = -u + I(t) - a(t)$$

com adaptação:

$$\tau_a \frac{da}{dt} = -a + \beta , \text{spike}(t)$$

### **Conexões do SNN**

O SNN recebe **e(t)** e propaga os potenciais:

$$\mathbf{h}(t) = \text{SNN}(\mathbf{e}(t))$$

Ele funciona como um **integrador neural**, aproximando soluções de EDOs.

## **Decoder – Mapeamento para Estado Orbital**

O vetor latente $\mathbf{h}(t)$ passa por um MLP:

$$[\hat{\mathbf{r}}(t), \hat{\mathbf{v}}(t)] = \text{MLP}_{dec}(\mathbf{h}(t))$$

Onde:

* $\hat{\mathbf{r}}(t) \in \mathbb{R}^3$
* $\hat{\mathbf{v}}(t) \in \mathbb{R}^3$

## **Physics Module (PINN-like)**

O componente PINN usa autograd para derivar $\hat{x}(t)$ e restringir sua dinâmica.

### **Derivadas via autograd**

$$\frac{d\hat{\mathbf{r}}}{dt} \approx \text{autograd}(\hat{\mathbf{r}}(t))$$

$$\frac{d\hat{\mathbf{v}}}{dt} \approx \text{autograd}(\hat{\mathbf{v}}(t))$$

### **Dinâmica orbital completa usada como alvo**

A aceleração real:

$$\mathbf{a}(\mathbf{r}, \mathbf{v}) = \mathbf{a}*{2B} + \mathbf{a}*{J2} + \mathbf{a}*{drag} + \mathbf{a}*{SRP}$$

#### **2-body**

$$\mathbf{a}_{2B} = -\frac{\mu}{\lVert \mathbf{r} \rVert^3} \mathbf{r}$$

#### **J2**

$$\mathbf{a}_{J2} = \frac{3 J_2 \mu R_e^2}{2 r^5}
\left[
\left( 1 - 5\frac{z^2}{r^2} \right)\mathbf{r} + 2 z \mathbf{k}
\right]$$

#### **Arrasto** (simplificado)

$$\mathbf{a}_{drag} = -\frac{1}{2} C_d \frac{A}{m} \rho(v) v \mathbf{v}$$

#### **SRP** (simplificado)

$$\mathbf{a}*{SRP} = P*{SRP} C_r \frac{A}{m} \hat{\mathbf{s}}$$


## **Loss Functions (Physics + Supervised + Measurement)**

A loss total:

$$\mathcal{L} = w_{meas} \mathcal{L}*{meas} + w*{dyn} \mathcal{L}*{dyn} + w*{energy} \mathcal{L}_{energy}$$

### **Measurement Loss**

Compara medições (RA/Dec ou range/Doppler):

$$\mathcal{L}_{meas} = \lVert \hat{y}(t) - y(t) \rVert^2$$

### **Physics Loss – Dinâmica**

$$\mathcal{L}_{dyn} = \left\lVert
\frac{d\hat{\mathbf{v}}}{dt} - \mathbf{a}(\hat{\mathbf{r}}, \hat{\mathbf{v}})
\right\rVert^2$$

### **Physics Loss – Energia**

Energia específica:

$$\epsilon = \frac{v^2}{2} - \frac{\mu}{r}$$

Com penalização:

$$\mathcal{L}_{energy} = (\epsilon(t) - \epsilon(t_0))^2$$

## **Fluxo de Treinamento**

1. Receber times $t_i$ 
2. Aplicar time encoding
3. Propagar no SNN
4. Decodificar para $\hat{\mathbf{r}}, \hat{\mathbf{v}}$
5. Calcular derivadas via autograd
6. Aplicar leis orbitais
7. Somar perdas
8. Backprop com surrogate gradients

## **Fluxo de Inferência (Tracking)**

Para qualquer $t$:

* gerar $e(t)$
* obter $\mathbf{h}(t)$
* decodificar estado
* filtrar com EKF