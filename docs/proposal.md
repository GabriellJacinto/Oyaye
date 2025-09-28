# Proposta de projeto: PINNs + SNNs para detecção/classificação de detritos

Como contribuição original, propomos um projeto que integra PINNs e SNNs em um sistema híbrido de detecção de detritos. A ideia central é usar SNNs processando dados de sensores neuromórficos (câmeras de eventos ou radares pulsados) para detectar e classificar objetos em tempo real, enquanto um módulo PINN garante consistência física das trajetórias inferidas. Um possível fluxo seria: (1) Gerar dados simulados de detritos orbitando (com estados verdadeiros conhecidos) e das leituras correspondentes de um sensor (imagens/eventos); (2) Treinar uma SNN (via SNNTorch/Nengo) para reconhecer padrões temporais de detritos nesses dados – por exemplo, identificando flashes ou trilhas nos eventos; (3) Paralelamente, treinar um PINN (via DeepXDE) que receba como entrada as estimativas de posição/velocidade oriundas da SNN ou dos próprios eventos, impondo as leis de Kepler e perturbadores de órbita na perda de treinamento. Esse PINN corrigiria as predições da SNN para que respeitem dinâmicas reais.

Como exemplo de arquitetura, imagine uma rede NP-SNN análoga à de Pham et al. – isto é, uma SNN cuja função de perda inclui um termo que penaliza desvios das EDOs orbitais (por exemplo, diferenças entre acelerações preditas e as calculadas pela gravidade de Newton). Alternativamente, pode-se treinar a SNN normalmente e depois usar o PINN para ajustar iterativamente sua saída (uma espécie de filtro físico). O sistema final permitiria, por exemplo, que um satélite detectasse um objecto menor e rapidamente estimasse sua órbita sem perder a coerência física.

Para desenvolver este trabalho em ~15 páginas, sugerimos os seguintes passos resumidos:

1. Dataset sintético: use Orekit/Poliastro para simular órbitas de detritos (e.g. fragmentos de um satélite), gerando posições/velocidades verdadeiras. Simule leituras de um sensor de eventos (usando PANGU ou um simulador de DVS customizado) correspondentes.

2. Treinamento SNN: com SNNTorch ou Nengo, treine uma SNN para classificar detritos em sequências de eventos ou imagens integradas. Avalie métricas de detecção (precisão, recall).

3. Treinamento PINN: usando DeepXDE, implemente um PINN que aprende a prever a próxima posição/velocidade do objeto sob leis de gravidade newtoniana e drag, ajustando à saída da SNN como “observação”. Compare erro de predição com/sem termo físico na perda.

4. Integração: combine os dois módulos num pipeline – por exemplo, faça fine-tuning em que a SNN corrente gera inputs para o PINN, e vice-versa. Experimente treiná-los em conjunto (backpropagando através dos dois) ou encadeá-los.

5. Avaliação: use conjuntos de teste simulados e, se possível, insira dados reais limitados (por exemplo, trajetórias conhecidas de satélites ou observações de telescópio) para validar robustez. Meça não só precisão, mas também latência e consumo energético simulado (benchmark básico).

Essa proposta segue lacunas na literatura identificadas: não há, até onde sabemos, um sistema neuromórfico-físico unificado para SSA. Acreditamos que combinar os ganhos de eficiência dos SNNs com a rigidez física dos PINNs produzirá um método inovador. Em um paper de ~15 páginas, descreveríamos a arquitetura proposta, os experimentos de simulação (incluindo conjuntos de dados gerados e métricas) e compararíamos com abordagens tradicionais (PINN puro, SNN puro). Bibliotecas como SNNTorch, DeepXDE e ferramentas de simulação (Orekit, PANGU/SPIN) garantirão viabilidade no prazo curto.