# **Proposta de Projeto – Framework Neuromórfico e Physics-Informed Neural Networks para Detecção e Rastreamento de Detritos Espaciais**

## **Visão Geral do Projeto**

Este projeto propõe o desenvolvimento de um framework modular que combina **Redes Neurais Espiking (SNNs)** e **Modelagem Informada por Física (PINNs)** para construir soluções avançadas de **detecção, rastreamento, classificação e mitigação de risco de colisão** envolvendo detritos espaciais em órbita terrestre.

O foco da **Fase 1** será um **Proof-of-Concept (POC)** que valida o componente central: um modelo **NP-SNN (Neural Physics-Informed Spiking Neural Network)** capaz de prever trajetórias com estabilidade física e robustez a observações esparsas.

A médio e longo prazo, o framework poderá ser ampliado para aplicações em:

* SSA terrestre (Ground-Based Space Situational Awareness)
* navegação relativa de satélites utilizando sensores neuromórficos
* tomada de decisão para mitigação e manobras
* análise de anomalias e fragmentações


## **Motivação e Relevância Aeroespacial**

O aumento acelerado do número de satélites e fragmentos em órbita tornou a detecção e o rastreamento de detritos um desafio crítico para:

* proteção de satélites operacionais;
* prevenção de colisões em LEO/GEO;
* redução do risco de cascata de Kessler;
* monitoramento ativo do ambiente espacial.

Além disso, abordagens tradicionais dependem de grandes volumes de observações e de propagadores determinísticos como SGP4, que apresentam grande drift em horizontes longos ou em órbitas perturbadas.

A combinação **SNN + PINN** oferece vantagens únicas:

* representação natural de dinâmicas contínuas;
* robustez a dados irregulares e ruidosos;
* capacidade de incorporar leis físicas diretamente na otimização;
* menor dependência de datasets massivos;
* caminho futuro para execução em hardware neuromórfico de baixo consumo.


## **Arquitetura Geral do Framework (Visão Modular)**

**A. Emulação e Ingestão de Sensores**
Fontes: telescópios ópticos (RA/Dec), radar (range/Doppler), câmeras event-based (futuro), simulação sintética.

**B. Pré‑processamento e Associação de Observações**
Limpeza, filtro de ruído, formação de tracklets e data association.

**C. Núcleo NP-SNN (SNN + PINN)**
Modelo central que aprende a dinâmica orbital contínua incorporando leis físicas (J2, arrasto, SRP).

**D. Filtro Híbrido (EKF/UKF/PF + NP‑SNN)**
Fusão de sensores e correção online.

**E. Classificação, Anomalias e Fragmentações**
Classificação de tipos de detritos, identificação de manobras e ruptura.

**F. Avaliação de Risco e Mitigação**
Cálculo de conjunção, probabilidade de colisão e recomendação de manobras.

**G. Deploy e Hardware Neuromórfico (Futuro)**
Integração com Loihi/TrueNorth, otimização energética, execução embarcada.

A **Fase 1** implementará apenas o módulo **C**, com suporte mínimo de **A e D**.

## **Escopo da Fase 1 (Proof-of-Concept)**

A Fase 1 é projetada para ser **viável**, **cientificamente relevante** e **avaliável pela banca**. Seus objetivos são:

### **Objetivos da Fase 1**

* Criar um gerador de dados sintéticos realistas para detritos em LEO.
* Implementar o modelo **NP-SNN** com:

  * codificação temporal contínua;
  * núcleo SNN com neurônios LIF/RLIF;
  * decoder para estado orbital (r, v);
  * perdas físicas (PINN-like): dinâmica, energia, invariantes.
* Comparar contra baselines clássicos: EKF e SGP4.
* Avaliar robustez a observações esparsas e horizonte de previsão.

### **Métricas de Sucesso**

* **RMSE de posição** nos horizontes: 10 min, 1 h, 6 h e 24 h.
* **Drift energético relativo**.
* **Erro sob observações esparsas** (ex.: drop de 30% das medições).
* **Comparação**: NP-SNN deve superar SGP4/EKF em pelo menos um cenário.
* Todos os experimentos registrados em MLflow/W&B.

### **Entregáveis da Fase 1**

1. **Dataset sintético documentado** + scripts de geração.
2. **Modelo NP-SNN funcional** com pipeline de treino e validação.
3. **Comparação quantitativa** com SGP4/EKF.
4. **Plots e gráficos**:

   * erro vs horizonte;
   * drift energético;
   * trajetórias previstas × verdade.
5. **Relatório técnico** da Fase 1.
6. **Repositório reprodutível** com:

   * README completo;
   * configs YAML;
   * scripts de treino e avaliação;
   * logging via MLflow.

## **Metodologia Técnica da Fase 1**

### **Geração de Dados Sintéticos**

* Implementação de um simulador orbital com:

  * gravidade J2;
  * arrasto atmosférico (NRLMSISE‑00, simplificado);
  * pressão de radiação solar (SRP);
  * órbitas circulares, elípticas e perturbadas.
* Adição de ruído realista para RA/Dec e range/Doppler.
* Exportação em formato padronizado para ingestão do modelo.

### **Arquitetura NP‑SNN**

* Time Encoding (Fourier/MLP).
* Núcleo SNN com neurônios LIF ou RLIF.
* Decoder para vetor de estado (6D).
* Physics loss:

  * L_dyn: discrepância entre derivada do modelo e aceleração física;
  * L_energy: penalização por desvio de invariantes;
  * L_meas: diferença entre previsões e observações.

### **Processo de Treinamento**

* Curriculum learning:

  * pretraining supervisionado em curtos horizontes;
  * transição gradual para perda física dominante.
* Collocation points entre observações.
* Otimização AdamW + gradiente surrogate para spikes.
* Logging integral de métricas.

## **Riscos e Mitigações**

**Instabilidade de treinamento SNN:** iniciar com supervisão; fallback para MLP-PINN.
**Gap sim→real:** domain randomization + TLE weak supervision.
**Carga computacional:** simplificar física se necessário; usar batch pequeno.

## **Impacto Esperado**

Esta proposta permitirá demonstrar a viabilidade de um modelo híbrido neuromórfico‑informado‑por‑física para aplicações aeroespaciais, criando o alicerce para um framework completo de SSA que poderá:

* funcionar com menos observações;
* prever órbitas de forma mais estável;
* detectar anomalias e fragmentações;
* integrar sensores neuromórficos no futuro.

O POC da Fase 1 já entrega valor científico e aplicado para justificar expansão futura.

## **Conclusão**

A Fase 1 fornece um POC robusto, reprodutível e tecnicamente sólido que valida o componente central do framework. O projeto equilibra inovação teórica (SNN+PINN), viabilidade prática (dataset sintético) e potencial de impacto para tecnologias aeroespaciais.