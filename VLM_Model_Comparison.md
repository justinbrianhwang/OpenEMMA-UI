# OpenEMMA VLM Model Comparison

## Overview

OpenEMMA uses a Chain-of-Thought (CoT) pipeline with a Vision-Language Model (VLM) for autonomous driving in CARLA 0.9.16. The VLM performs 4-step reasoning: Scene Description → Critical Objects → Driving Intent → Motion Prediction.

The VLM serves as an **advisory system** — actual vehicle control uses route-based pure-pursuit steering. The VLM's scene understanding is displayed in the UI and can modulate speed via intent.

Four VLM backends were tested on **Town01, same route, same traffic conditions**.

---

## 1. Model Specifications

| | LLaVA-v1.5-7b | LLaMA-3.2-11B-Vision | Qwen2-VL-7B | GPT-4o (API) |
|---|---|---|---|---|
| **Parameters** | 7B | 11B | 7B | Unknown (closed) |
| **Type** | Local | Local | Local | Cloud API |
| **VRAM (fp16)** | ~14 GB | ~22 GB | ~16 GB | 0 GB |
| **Conda Env** | `openemma` or `avllm` | `openemma` | `openemma` | `openemma` |
| **transformers** | 4.36.2+ | 4.46.2+ | 4.46.2+ | N/A |
| **Cost** | Free | Free | Free | ~$0.01/frame |

---

## 2. Quantitative Performance (GPT-4o — 4,341 frames)

> Note: 정량 데이터는 GPT-4o debug CSV에서 추출 (다른 모델은 CSV 미보존으로 CoT 로그 기반 정성 비교)

| Metric | Value |
|---|---|
| Total frames | 4,341 |
| Avg speed | 8.9 km/h |
| Max speed | 17.0 km/h |
| Zero speed frames | 1,279 / 4,341 (29.5%) |
| Avg absolute steer | 0.0377 |
| Brake frames | 31 / 4,341 (0.7%) |
| VLM "red light" in scene | 453 / 4,217 (10.7%) |
| VLM "stop" intent | 1,367 / 4,120 (33.2%) |
| VLM speed = 0 | 1,360 / 4,103 (33.1%) |

**GPT-4o Intent Distribution:**
| Intent | Count | % |
|---|---|---|
| Go straight and maintain speed | 2,142 | 52.0% |
| Go straight and stop | 1,207 | 29.3% |
| Go straight and slow down | 311 | 7.5% |
| Go straight and speed up | 300 | 7.3% |
| Other | 160 | 3.9% |

---

## 3. Qualitative CoT Comparison

### 3.1 Scene Description (CoT Step 1)

| Model | Quality | Example |
|---|---|---|
| **LLaVA** | Poor | "The driving scene in the virtual city includes traffic lights, other vehicles, pedestrians, and lane markings." (매 프레임 동일) |
| **LLaMA** | Good | "A virtual city street with a long, straight road and no other vehicles." / "A yellow line down the center..." |
| **Qwen** | Decent | "A straight road with yellow lane markings. The road is wide..." |
| **GPT-4o** | Excellent | "A straight road with a wet surface reflecting the setting sun." / "Double yellow lane markings, indicating two-way traffic." |

**분석:**
- LLaVA: 고정 템플릿 반복 → 실제 장면 인식 불가
- LLaMA: 프레임별로 다른 설명, 도로 형태/차선/차량 유무 식별
- Qwen: LLaVA보다 나으나 변화 적음
- GPT-4o: 노면 상태, 태양 반사, 이중 실선 등 미세 디테일까지 포착

### 3.2 Critical Objects (CoT Step 2)

| Model | Hallucination Rate | Primary Issue |
|---|---|---|
| **LLaVA** | ~95%+ | **항상** "Red traffic light" hallucination |
| **LLaMA** | ~5% | 가끔 불필요 객체 언급, 대체로 정확 |
| **Qwen** | ~70% | **빈번한** "Red Traffic Light" hallucination |
| **GPT-4o** | ~10% | 실제 신호등만 감지, 노면/표지판/장벽 등 다양 |

**핵심 발견:**
- LLaVA & Qwen: "Red traffic light" bias가 학습 데이터에 강하게 각인 → 빈 도로에서도 빨간불 보고
- LLaMA: hallucination 가장 적음 (로컬 모델 중)
- GPT-4o: hallucination 최소, 다양한 객체 카테고리 인식

### 3.3 Driving Intent (CoT Step 3)

| Model | Stop 비율 (추정) | Intent 다양성 | 정확도 |
|---|---|---|---|
| **LLaVA** | ~5% | 매우 낮음 ("Go straight" 반복) | Low |
| **LLaMA** | ~15% | 높음 (speed up / maintain / slow down) | High |
| **Qwen** | ~60% | 낮음 ("Stop" 과다) | Low |
| **GPT-4o** | ~33% | 높음 (maintain 52% / stop 29% / slow 8% / speed up 7%) | High |

**분석:**
- Qwen: red light hallucination → 60%+ "Stop" → 주행 속도 저하
- GPT-4o: 4가지 intent 균형 있게 분포, 실제 상황 반영
- LLaMA: GPT-4o와 유사한 패턴, 상황 인식 기반 intent

### 3.4 Motion Prediction (CoT Step 4)

| Model | Format | 파싱 성공률 | Speed Profile |
|---|---|---|---|
| **LLaVA** | 불안정 | ~60% | 수치 hallucination 빈번 |
| **LLaMA** | `5,0; 5.5,0; 6,0` | ~95% | 점진적 가속 |
| **Qwen** | `0,0; 0,0; 0,0` | ~90% | 자주 0 (Stop 영향) |
| **GPT-4o** | `3.8,0; 3.8,0; 3.8,0` | ~98% | 가속/감속 프로파일 (1.5→1.0→0.0) |

---

## 4. Overall Summary

| Criteria | LLaVA-v1.5-7b | LLaMA-3.2-11B | Qwen2-VL-7B | GPT-4o |
|---|---|---|---|---|
| **Scene Quality** | ★☆☆☆☆ | ★★★★☆ | ★★★☆☆ | ★★★★★ |
| **Hallucination** | ★☆☆☆☆ (severe) | ★★★★☆ (minimal) | ★★☆☆☆ (moderate) | ★★★★★ (minimal) |
| **Intent Accuracy** | ★★☆☆☆ | ★★★★☆ | ★★☆☆☆ | ★★★★★ |
| **Motion Quality** | ★★☆☆☆ | ★★★★☆ | ★★★☆☆ | ★★★★★ |
| **Inference Speed** | ★★★★★ (~2s) | ★★★☆☆ (~4s) | ★★★★☆ (~3s) | ★★★★★ (~1-2s) |
| **VRAM** | ★★★★☆ (14GB) | ★★★☆☆ (22GB) | ★★★★☆ (16GB) | ★★★★★ (0GB) |
| **Cost** | ★★★★★ | ★★★★★ | ★★★★★ | ★★☆☆☆ |
| **Overall** | ★★☆☆☆ | ★★★★☆ | ★★★☆☆ | ★★★★★ |

---

## 5. Ranking & Recommendation

| Rank | Model | Strengths | Weaknesses |
|---|---|---|---|
| 1 | **GPT-4o** | 최고 scene 이해, hallucination 최소, 점진적 motion planning | API 비용 (~$0.01/frame), 인터넷 필요 |
| 2 | **LLaMA-3.2-11B** | 로컬 최강, 낮은 hallucination, 다양한 intent | VRAM 22GB 필요, 추론 속도 느림 |
| 3 | **Qwen2-VL-7B** | 괜찮은 scene 묘사, 깔끔한 포맷 | Red light hallucination → Stop 과다 |
| 4 | **LLaVA-v1.5-7b** | 가볍고 빠름 | 모든 영역에서 hallucination, generic 응답 |

**연구 권장사항:**
- **논문 실험용**: GPT-4o (최고 성능 baseline) + LLaMA-3.2-11B (최고 로컬 모델) 병행
- **실시간 데모**: LLaMA-3.2-11B (무료, VRAM 충분)
- **비용 제한**: LLaMA-3.2-11B 단독 사용

---

## 6. Run Commands

```bash
# GPT-4o (API — best quality)
OPENAI_API_KEY=sk-... conda run -n openemma python openemmaUI.py --gpt

# LLaMA-3.2-11B (local — best free option)
conda run -n openemma python openemmaUI.py --llama

# Qwen2-VL-7B (local)
conda run -n openemma python openemmaUI.py --qwen

# LLaVA-v1.5-7b (local — not recommended)
conda run -n openemma python openemmaUI.py --llava
```

## 7. Notes
- 모든 모델은 동일한 route-based pure-pursuit controller로 주행 (VLM은 advisory)
- `openemma` conda env (transformers 5.3.0+)에서 4개 모델 모두 지원
- `avllm` env (transformers 4.36.2)는 LLaVA만 지원
- GPT-4o는 `OPENAI_API_KEY` 환경변수 필요
- 정량 비교를 위해 각 모델별 debug CSV 별도 저장 기능 추가 필요 (TODO)