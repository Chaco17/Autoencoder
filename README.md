# Autoencoder

Tarea 03: generación/reconstrucción de imágenes con un VAE y un autoencoder U-Net sobre MVTec AD,
usando PyTorch Lightning, Hydra y Weights & Biases.

## Estructura

```
conf/        Configuraciones de Hydra (config.yaml, model/, trainer/, logger/, data/, loss/)
src/         Código: dataset, modelos, pérdidas, LightningModule y callbacks de W&B
notebooks/   Notebook de entrega paso a paso
data/        Dataset (no se sube a git)
reports/     Figuras exportadas de W&B
train.py     Punto de entrada del entrenamiento
```

## Entorno

Entorno virtual con Python 3.11 (desde la raíz del repo, PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m ipykernel install --user --name autoencoder --display-name "Python (.venv Autoencoder)"
.venv\Scripts\wandb login
```

Después, abrir `notebooks/main.ipynb` con el kernel `.venv` y ejecutarlo en orden (el entrenamiento se lanza
desde la sección 6 del notebook).

## Experimentos

Cada combinación modelo (VAE, U-Net) × pérdida (L1, L2, SSIM, SSIM+L1) se entrena con varios **trials**
(conjuntos de hiperparámetros en `conf/trial/`: `base`, `e200_cos`, `e200_cos_z256`): 3 × 8 = 24 corridas.

```powershell
.venv\Scripts\python train.py -m                        # todos los trials x 2 modelos x 4 pérdidas
.venv\Scripts\python train.py -m trial=e200_cos         # un solo trial (8 corridas)
.venv\Scripts\python train.py model=u-net loss=ssim trial=e200_cos_z256
```

Las corridas que ya existen **se saltan** (no se duplican en W&B). Esto incluye las guardadas en `outputs/`/`multirun/`
y las corridas terminadas en el proyecto de W&B, comparando un hash de sus hiperparámetros (`src/registry.py`).
Para forzar un reentrenamiento: `skip_existing.enabled=false`.

### Histogramas del error en W&B

Después de entrenar, `scripts/log_histograms.py` agrega a cada corrida terminada sus histogramas del error
(imágenes buenas vs cada tipo de defecto, por clase y puntaje) a partir de su tabla `test/scores`, sin reentrenar.
Las corridas que ya los tienen se saltan. No ejecutarlo mientras corre `train.py` o el notebook.

```powershell
.venv\Scripts\python scripts/log_histograms.py --dry-run          # solo lista lo que haría
.venv\Scripts\python scripts/log_histograms.py                    # todas las corridas
.venv\Scripts\python scripts/log_histograms.py --trial e200_cos   # solo un trial
```

## Dataset

Se usan solo 4 clases de MVTec AD: `cable`, `capsule`, `screw` y `transistor`.

1. Descargar desde la página oficial (https://www.mvtec.com/company/research/datasets/mvtec-ad)
   los archivos `cable.tar.xz`, `capsule.tar.xz`, `screw.tar.xz` y `transistor.tar.xz` (~1.5 GB en total).
2. Desde la raíz del repo, extraerlos en `data/mvtec_ad/` (cambiar la ruta de descarga si es distinta):

   ```bash
   mkdir -p data/mvtec_ad
   for c in cable capsule screw transistor; do tar -xJf ~/Downloads/$c.tar.xz -C data/mvtec_ad; done
   ```

3. Debe quedar así:

   ```
   data/mvtec_ad/
   ├── cable/
   │   ├── train/good/*.png          # solo imágenes buenas (entrenamiento + validación)
   │   ├── test/good/*.png           # imágenes buenas de prueba
   │   ├── test/<tipo_defecto>/*.png # imágenes con anomalías (bent_wire, cut_lead, ...)
   │   └── ground_truth/             # máscaras de defectos (no se usan)
   ├── capsule/
   ├── screw/
   └── transistor/
   ```

   | Clase      | train/good | test (total) |
   |------------|-----------:|-------------:|
   | cable      | 224        | 150          |
   | capsule    | 219        | 132          |
   | screw      | 320        | 160          |
   | transistor | 213        | 100          |

Las imágenes originales son de 1024x1024 (`screw` en escala de grises); se redimensionan a 128x128 RGB
al cargarlas, así que no hace falta preprocesarlas a mano.
