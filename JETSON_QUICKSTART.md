# 🚀 Quick Start: Jetson Optimization

## Step 1: Convert Models to TensorRT (RTX 4090)

Na sua máquina com RTX 4090, rode:

```bash
cd /home/cvrr/Code/workzone
source venv/bin/activate

# Converter TODOS os modelos .pt para TensorRT
python scripts/optimize_for_jetson.py

# OU converter um modelo específico
python scripts/optimize_for_jetson.py --model weights/yolo12s_hardneg_1280.pt
```

**Output esperado:**
```
==================================================================
YOLO TensorRT Converter
==================================================================
✓ GPU: NVIDIA GeForce RTX 4090
✓ Memory: 24.0 GB
✓ CUDA Version: 12.8

==================================================================
Exporting: yolo12s_hardneg_1280.pt
==================================================================
Loading YOLO model...
  Precision: FP16
  Image size: 1280
  Workspace: 4GB

✅ Export successful!
   Engine: weights/yolo12s_hardneg_1280.engine
   Size: 45.2 MB

🔥 Benchmarking...
   Warming up...
   Running 100 iterations...

   Results:
   ⏱️  Mean: 2.1ms
   📊 FPS: 476.2
   📈 Throughput: 779.5 Mpix/s
```

## Step 2: Transfer para Jetson

Copie os arquivos `.engine` para o Jetson:

```bash
# Na máquina local (RTX 4090)
scp weights/*.engine jetson@<jetson-ip>:/path/to/workzone/weights/

# OU use rsync
rsync -avz weights/*.engine jetson@<jetson-ip>:/path/to/workzone/weights/
```

## Step 3: Rodar no Jetson

No Jetson Orin:

```bash
# Ativar modo de máxima performance
sudo nvpmodel -m 0
sudo jetson_clocks

# Rodar aplicação
cd /path/to/workzone
source venv/bin/activate
streamlit run src/workzone/apps/streamlit/app_phase2_1_evaluation.py
```

**O app detectará automaticamente os `.engine` files!**

## Verificar que TensorRT está sendo usado

No terminal do Streamlit, você verá:

```
🚀 TensorRT engine found: yolo12s_hardneg_1280.engine
✓ Loaded TensorRT model (optimized for Tensor Cores)
```

## Performance Esperada

### RTX 4090
- **Antes (.pt FP32)**: ~8ms/frame (125 FPS)
- **Depois (.engine FP16)**: ~2ms/frame (500 FPS)
- **Speedup**: 4x faster! ⚡

### Jetson Orin 64GB
- **Antes (.pt FP32)**: ~60ms/frame (16 FPS)
- **Depois (.engine FP16)**: ~15ms/frame (66 FPS)  
- **Speedup**: 4x faster! ⚡

## Opções Avançadas

### INT8 Precision (Máxima velocidade)

```bash
python scripts/optimize_for_jetson.py --int8 --model weights/yolo12s_hardneg_1280.pt
```

**Requer**: Dataset de calibração (~100-500 imagens)

**Speedup adicional**: 2-3x sobre FP16

### Tamanho de Imagem Menor

```bash
# Processar em 640x640 ao invés de 1280x1280 (4x mais rápido)
python scripts/optimize_for_jetson.py --imgsz 640
```

### FP32 (Máxima precisão, mais lento)

```bash
python scripts/optimize_for_jetson.py --fp32
```

## Troubleshooting

### "TensorRT engine not compatible"
- **Causa**: Engine foi criado em GPU diferente
- **Solução**: Re-gerar engine no Jetson:
  ```bash
  # No Jetson
  python scripts/optimize_for_jetson.py
  ```

### "Out of memory"
- **Causa**: Jetson tem menos memória
- **Solução 1**: Usar imgsz menor (640)
- **Solução 2**: Desabilitar CLIP ou OCR
- **Solução 3**: Processar frames em batch menor

### "Slow performance on Jetson"
- **Check 1**: Modo de performance ativo?
  ```bash
  sudo nvpmodel -q  # Should show mode 0
  ```
- **Check 2**: TensorRT being used?
  - Look for "🚀 TensorRT engine found" in logs
- **Check 3**: Temperature throttling?
  ```bash
  tegrastats
  ```

## Monitoring

### GPU Usage (Jetson)
```bash
tegrastats
```

### GPU Usage (RTX 4090)
```bash
nvidia-smi -l 1
```

### Profiling
```bash
# Detailed profiling
nsys profile python your_script.py
```

## Next Steps

1. ✅ Convert models with `optimize_for_jetson.py`
2. ✅ Verify TensorRT detection in app logs
3. ⏩ Benchmark FPS improvement
4. ⏩ Fine-tune for your workload
5. ⏩ Deploy to production!