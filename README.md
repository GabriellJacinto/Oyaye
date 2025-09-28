# Oyaye
Detecção e classificação de detritos espaciais com SNN + PINN.

## Quickstart
1. Clone:
   ```bash
   git clone https://github.com/usuario/ogum-sentinel.git
   cd Oyaye

    Setup (virtualenv/pip):

    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt

## Demo

Rode o notebook notebooks/01-data-generation.ipynb ou:
```
python src/sim/simulator.py --config configs/example.yml
python src/train.py --config configs/snn_small.yml
```

## Cite
<!-- Adicione aqui o bloco CITATION.cff ou o DOI do paper/preprint. -->

## LICENSE
<!-- Escolha: **Apache-2.0** (bom para research + indústria) ou **MIT** (mais permissiva).  
Exemplo header (Apache-2.0) — gere o arquivo completo via site choosealicense ou copie padrão. -->