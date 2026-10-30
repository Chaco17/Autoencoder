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
