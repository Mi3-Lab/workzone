# 🧪 Phase 1.4 Testing Guide

## ✅ Dataset Status

Os dados foram preparados com sucesso:
```
highway:   542 images
urban:   4,776 images  
suburban:  100 images (replicadas de highway)
parking:   100 images (replicadas de urban)
────────────────────
Total:   5,518 images
```

## 🚀 Como Testar

### Opção 1: Treino Completo (Recomendado)

```bash
cd /home/wesleyferreiramaia/data/workzone

# Submete para GPU (10 epochs, ~30-40 min no A100)
srun --gpus=1 --partition gpu -t 180 bash -c '
  source .venv/bin/activate
  python scripts/train_scene_context.py --epochs 10 --batch-size 32
' &

# Acompanha o treinamento
sleep 10
squeue -u wesleyferreiramaia
```

Esperado:
- ~2-3 min por epoch em A100
- Val accuracy: ~90%+ (highway/urban têm dados reais)
- Modelo salvo em: `weights/scene_context_classifier.pt`

---

### Opção 2: Teste Rápido (2 epochs)

```bash
cd /home/wesleyferreiramaia/data/workzone

srun --gpus=1 --partition gpu -t 60 bash -c '
  source .venv/bin/activate
  python scripts/train_scene_context.py --epochs 2 --batch-size 32
' &

# Acompanha
sleep 20
tail -100 /tmp/train_output.txt
```

Tempo: ~5-10 min total

---

### Opção 3: Teste em Video (sem Treino)

Usa o modelo MobileNetV2 pré-treinado (ImageNet):

```bash
cd /home/wesleyferreiramaia/data/workzone

python scripts/process_video_fusion.py data/demo/boston_short.mp4 \
  --enable-phase1-4 \
  --enable-phase1-1 \
  --no-motion \
  --no-video \
  --quiet
```

Resultado esperado:
```
Processing: boston_short.mp4
Phase 1.4 Scene Context: urban (conf: 0.95)
APPROACHING: 12 frames
OUT: 45 frames
Time: 23.5s
```

---

## 📊 Verificar Treinamento

### Durante o treinamento:
```bash
# Ver últ imas linhas do output
tail -50 /tmp/train_output.txt

# Ou acompanhar em tempo real
tail -f /tmp/train_output.txt
```

### Após completar:
```bash
ls -lh weights/scene_context_classifier.pt

# Esperado: ~13 MB
```

---

## 🔧 Caso os Symlinks Quebrarem

Se receber erro tipo `FileNotFoundError: pgh04_0458.JPG`, execute:

```bash
cd /home/wesleyferreiramaia/data/workzone
python3 << 'EOF'
import os
from pathlib import Path

dataset_dir = Path("/data/wesleyferreiramaia/workzone/data/04_derivatives/scene_context_dataset")
images_base = Path("/data/wesleyferreiramaia/workzone/data/01_raw/images")

for context in ["highway", "urban", "suburban", "parking"]:
    context_dir = dataset_dir / context
    for link in context_dir.iterdir():
        if link.is_symlink():
            filename = Path(os.readlink(link)).name
            abs_target = images_base / filename
            if abs_target.exists():
                os.unlink(link)
                os.symlink(abs_target, link)
                
print("✓ Symlinks fixed!")
EOF
```

---

## 📈 Próximos Passos

1. **Treinar com mais dados** (opcional)
   - Se houver mais anotações com scene_level_tags, adicione a suburban/parking

2. **Avaliar em vídeos** 
   - Compare resultados com/sem Phase 1.4
   - Medir redução de FP

3. **Fine-tune thresholds**
   - Editar `src/workzone/models/scene_context.py`
   - Ajustar `SceneContextConfig.THRESHOLDS`

4. **Produção**
   - Commit do modelo treinado
   - Deploy com `--enable-phase1-4`

---

## ⚠️ Troubleshooting

| Erro | Solução |
|------|---------|
| `FileNotFoundError: pgh04_0458.JPG` | Executar fix de symlinks acima |
| `RuntimeError: CUDA not found` | Usar `srun --gpus=1` ou --no-gpu |
| `OOM: out of memory` | Reduzir `--batch-size` (padrão: 32) |
| Job stuck | Cancelar com `scancel <JOBID>` |

---

## 📝 Log Padrão (esperado)

```
INFO:__main__:Training scene context classifier
INFO:__main__:  Dataset: 5518 images (4414 train, 1104 val)
INFO:__main__:  Contexts: ['highway', 'urban', 'suburban', 'parking']
INFO:__main__:  Epochs: 10, Batch size: 32, LR: 0.001
INFO:__main__:Epoch 1/10 | Train Loss: 1.2345, Acc: 45.2% | Val Loss: 1.1234, Acc: 52.3%
  ✓ Saved best model (val_acc: 52.3%)
INFO:__main__:Epoch 2/10 | Train Loss: 0.8765, Acc: 68.1% | Val Loss: 0.7654, Acc: 71.2%
  ✓ Saved best model (val_acc: 71.2%)
...
INFO:__main__:✅ Training complete! Best val accuracy: 91.5%
INFO:__main__:   Weights saved to: weights/scene_context_classifier.pt
```

---

**Status:** Dataset preparado ✅  
**Próximo:** Treinar modelo ou testar em vídeo  
**Tempo estimado:** 30-40 min (treino completo)
