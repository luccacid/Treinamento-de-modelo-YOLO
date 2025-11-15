# Documentação do Pipeline YOLO — Organização, Treinamento, Validação e Limpeza

Este documento descreve o funcionamento e o uso dos quatro scripts do pipeline para preparar, treinar e validar modelos YOLO com datasets customizados.

---

# `01_organize_dataset.py`

### Função

Organizar automaticamente a estrutura do dataset no formato esperado pelo YOLO:

```
datasets/
 ├── treino/
 │    ├── images/
 │    └── labels/
 ├── validacao/
 │    ├── images/
 │    └── labels/
 └── teste/
      ├── images/
      └── labels/
```

### O que o script faz

* Move imagens (`.jpg`, `.jpeg`, `.png`) para a pasta `images/`
* Move arquivos de labels (`.txt`) para a pasta `labels/`
* Cria as pastas necessárias, caso não existam
* Ignora arquivos que não sejam imagens ou labels

### Como executar

```bash
python 01_organize_dataset.py --path datasets
```

### Argumentos

| Argumento | Padrão     | Descrição                                         |
| --------- | ---------- | ------------------------------------------------- |
| `--path`  | `datasets` | Caminho contendo as pastas treino/validacao/teste |

---

# `02_train_model.py`

### Função

Treinar modelos YOLO (v8/v9) usando a biblioteca Ultralytics.

### O que o script faz

* Detecta automaticamente GPU (CUDA) ou utiliza CPU
* Carrega o modelo base informado
* Executa o treinamento com parâmetros configuráveis
* Salva os resultados em `runs/detect/<name>/`

### Como executar

```bash
python 02_train_model.py \
  --model yolov9m.pt \
  --data data/dataset.yaml \
  --epochs 100 \
  --batch 8 \
  --imgsz 640 \
  --name placa_v9m
```

### Argumentos

| Argumento  | Padrão              | Descrição                              |
| ---------- | ------------------- | -------------------------------------- |
| `--model`  | yolov9m.pt          | Modelo base ou checkpoint              |
| `--data`   | data/dataset.yaml   | Caminho para o arquivo YAML do dataset |
| `--epochs` | 100                 | Número de épocas                       |
| `--batch`  | 8                   | Tamanho do lote                        |
| `--imgsz`  | 640                 | Tamanho das imagens                    |
| `--name`   | license_plate_train | Nome do experimento dentro de `runs/`  |

---

# `03_validate_model.py`

### Função

Validar um modelo YOLO treinado para gerar métricas como mAP, precisão e recall.

### O que o script faz

* Detecta automaticamente GPU ou CPU
* Carrega o modelo informado
* Executa a validação com os parâmetros fornecidos
* Salva os resultados em `runs/val/<name>/`

### Como executar

```bash
python 03_validate_model.py \
  --model runs/detect/placa_v9m/weights/best.pt \
  --data data/dataset.yaml \
  --batch 16 \
  --imgsz 640 \
  --name placa_v9m_val
```

### Argumentos

| Argumento | Obrigatório | Descrição                                  |
| --------- | ----------- | ------------------------------------------ |
| `--model` | Sim         | Caminho para `best.pt` ou outro checkpoint |
| `--data`  | Não         | Caminho para o YAML do dataset             |
| `--batch` | Não         | Tamanho do lote                            |
| `--imgsz` | Não         | Tamanho das imagens                        |
| `--name`  | Não         | Nome da pasta de resultados                |

---

# `04_clear_dataset.py`

### Função

Limpar arquivos do dataset e remover caches criados pelo YOLO.

### O que o script faz

* Deleta imagens (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`)
* Deleta labels `.txt`
* Remove todos os arquivos `.cache` dentro do dataset
* Oferece modo de simulação (dry-run)
* Permite pular confirmações com o parâmetro `-y`

### Como executar

Simulação (não deleta nada):

```bash
python 04_clear_dataset.py --dry-run
```

Limpeza real:

```bash
python 04_clear_dataset.py -y
```

### Argumentos

| Argumento     | Descrição                                |
| ------------- | ---------------------------------------- |
| `--path`      | Caminho da pasta base do dataset         |
| `--dry-run`   | Apenas mostra o que seria removido       |
| `-y`, `--yes` | Executa a exclusão sem pedir confirmação |
Aqui está a **seção adicional em Markdown** explicando o `dataset.yaml`, no mesmo estilo do seu documento atual (sem emojis, direto, técnico e consistente):

---

# `dataset.yaml`

### Função

Definir os caminhos do dataset e a lista de classes usadas pelo YOLO durante o treinamento, validação e teste.

### Estrutura do arquivo

```yaml
# Caminhos relativos ao arquivo YAML (precisa voltar um nível)
train: ../datasets/treino/images
val: ../datasets/validacao/images
# test: ../datasets/teste/images

# Definições das classes
nc: 35
names:
  [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
    'V', 'W', 'X', 'Y', 'Z' ]
```

---

## Campos

### `train`

```
train: ../datasets/treino/images
```

Caminho para a pasta contendo as imagens de treino.
Os arquivos de label correspondentes devem estar em:

```
../datasets/treino/labels
```

---

### `val`

```
val: ../datasets/validacao/images
```

Caminho das imagens usadas na validação do modelo.
Labels devem estar em:

```
../datasets/validacao/labels
```

---

### `test` (opcional)

```
# test: ../datasets/teste/images
```

Caminho das imagens de teste.
Rotulado como opcional e, por isso, comentado.
Se ativado, os labels devem estar em:

```
../datasets/teste/labels
```

---

### `nc`

```
nc: 35
```

Número total de classes do dataset.
Deve corresponder exatamente ao total de itens em `names`.

---

### `names`

Lista das classes usadas nos arquivos de label `.txt`.

A lista indica qual string textual corresponde ao índice numérico usado nos labels YOLO.

Exemplo:
Se um arquivo `.txt` tiver a classe `4`, ela se refere a:

```
names[4] → '4'
```

A ordem é obrigatória e não pode ser alterada sem ajustar os labels.
