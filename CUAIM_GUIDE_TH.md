# คู่มือรันโปรเจกต์ GABM บน CU-AIM

คู่มือนี้พาตั้งแต่ดึงโค้ดจาก GitHub ไปจนได้ผลลัพธ์กลับมาที่เครื่องตัวเอง
ใช้คู่กับ *CU-AIM Server User Manual 2026* (ในคู่มือนี้อ้างเป็น "คู่มือ server บทที่ x")

**งานที่ server ทำให้:**
- รัน notebook 02 (feasibility), 04 และ 05 แบบไม่ต้องเปิด notebook
- งาน GPU 1 งานต่อ 1 โมเดล แต่ละงานรัน condition C และ D โมเดลละ 3 seeds ที่ 180 วัน
- เมื่อโมเดลครบ จะมีงาน CPU อีก 1 งานรัน analysis ของ notebook 04/05 แล้วบันทึกตาราง กราฟ และ `metrics.csv`

**ทำไมผลถึงใช้ร่วมกับ Colab ได้:** `scripts/run_server.py` เรียก `run_simulation` แบบเดียวกับใน notebook ทุกตัว
- **run_id ตรงกับที่ได้จาก Colab ทุกตัว**
- ผลจาก server จึงเปิดดูใน notebook บน Colab หรือบนเครื่องได้ทันที
- run ที่เคยรันเสร็จบน Colab แล้ว server จะข้ามให้เอง

---

## สารบัญ

0. [สิ่งที่ต้องมีก่อนเริ่ม](#0-สิ่งที่ต้องมีก่อนเริ่ม)
1. [บนเครื่องตัวเอง: push โค้ดขึ้น GitHub](#1-บนเครื่องตัวเอง-push-โค้ดขึ้น-github)
2. [เข้า CU-AIM](#2-เข้า-cu-aim)
3. [clone โปรเจกต์](#3-clone-โปรเจกต์)
4. [ตรวจค่าใน server.env](#4-ตรวจค่าใน-serverenv)
5. [(ทางเลือก) อัปโหลดผลเดิมจาก Colab](#5-ทางเลือก-อัปโหลดผลเดิมจาก-colab)
6. [ติดตั้ง environment (ครั้งเดียว)](#6-ติดตั้ง-environment-ครั้งเดียว)
7. [ใส่ HF token และดาวน์โหลดโมเดล](#7-ใส่-hf-token-และดาวน์โหลดโมเดล)
8. [Smoke test](#8-smoke-test)
9. [รันจริง](#9-รันจริง)
10. [ติดตามงาน / ยกเลิกงาน](#10-ติดตามงาน--ยกเลิกงาน)
11. [ดึงผลกลับมาที่เครื่อง](#11-ดึงผลกลับมาที่เครื่อง)
12. [อัปเดตโค้ดรอบถัดไป (git pull)](#12-อัปเดตโค้ดรอบถัดไป-git-pull)
13. [ปรับแต่งการรัน](#13-ปรับแต่งการรัน)
14. [แก้ปัญหาที่อาจเจอ](#14-แก้ปัญหาที่อาจเจอ)
15. [ไฟล์ใน scripts/server ทำอะไร](#15-ไฟล์ใน-scriptsserver-ทำอะไร)

---

## 0. สิ่งที่ต้องมีก่อนเริ่ม

- [ ] บัญชี CU-AIM และ SSH key ที่ admin อนุมัติแล้ว (คู่มือ server บทที่ 1–2)
- [ ] เชื่อม Chula VPN ได้
- [ ] รู้ path ของ project storage ของกลุ่ม เช่น `/data/project/<กลุ่ม>/<โฟลเดอร์ของเรา>` (คู่มือ server 3.1)
  - **ห้ามใช้ `/data/home`** เพราะโควตาเล็ก แต่โมเดลรวมกันประมาณ 80 GB และ environment อีกหลาย GB
- [ ] บัญชี Hugging Face และ token แบบ **Read**
- [ ] กด accept licence ของ `meta-llama/Llama-3.1-8B-Instruct` บนหน้า huggingface.co แล้ว
  - llama8b เป็น gated model ส่วนโมเดลอื่นไม่ต้องใช้ token
- [ ] พื้นที่ว่างใน project storage อย่างน้อยประมาณ 120 GB
  - โมเดล ~80 GB, container + venv ~10 GB, และผลลัพธ์

---

## 1. บนเครื่องตัวเอง: push โค้ดขึ้น GitHub

server จะ `git clone` โค้ดจาก GitHub ดังนั้นโค้ดล่าสุด (รวม `scripts/server/`) ต้องอยู่บน GitHub ก่อน

```powershell
cd C:\Users\USER\Documents\bioinfo\gabm
git status                     # ดูว่ามีไฟล์อะไรเปลี่ยนบ้าง
git add .gitattributes .gitignore README.md CUAIM_GUIDE_TH.md config gabm notebooks requirements-server.txt scripts
git commit -m "Add CU-AIM SLURM setup and model panel"
git push origin main
```

- **`outputs/` ไม่อยู่ใน git แล้ว** (ตั้งไว้ใน `.gitignore`)
  - clone ไปบน server จะได้โฟลเดอร์ผลลัพธ์ว่าง และ server จะรันทุกอย่างใหม่หมด รวม A/B baseline ที่ notebook 04 สร้างเองบน CPU
  - ผลบนเครื่องเรายังอยู่ครบ
- **`survival-manual-cuaim.pdf` ถูก ignore ไว้** เพราะในไฟล์มี username ของคนอื่นและที่อยู่ IP ของ server

---

## 2. เข้า CU-AIM

1. เชื่อม Chula VPN
2. ssh เข้า login node ตามคู่มือ server บทที่ 2:

```bash
ssh <username>@<ที่อยู่ CU-AIM ตามคู่มือ>
```

ทุกคำสั่งจากนี้ไปรันบน **login node**
- ไม่ต้องขอ GPU เอง เพราะสคริปต์ส่งงานเข้า SLURM ให้
- งานที่ต้องใช้ GPU จะรันบน compute node

---

## 3. clone โปรเจกต์

```bash
cd /data/project/<กลุ่ม>/<โฟลเดอร์ของเรา>
git clone https://github.com/Professor-Shin/Epidemic-Agent-for-Thai.git gabm
cd gabm
```

**ถ้า repo เป็น private** git จะถามรหัสผ่าน
- ให้ใส่ GitHub **Personal Access Token** แทนรหัสผ่าน
  - สร้างได้ที่ GitHub → Settings → Developer settings → Fine-grained tokens โดยให้สิทธิ์ Contents: Read-only
- หรือใช้ SSH key ของ server ไปเพิ่มใน GitHub แล้ว clone ด้วย `git@github.com:Professor-Shin/Epidemic-Agent-for-Thai.git`

**ไม่ต้องดาวน์โหลดข้อมูลเพิ่ม:** ข้อมูลที่ simulation ใช้ (`data/processed/`, `config/personas.csv`) อยู่ใน git แล้ว

---

## 4. ตรวจค่าใน server.env

ทุกสคริปต์อ่านค่าจาก `scripts/server/server.env`
- ค่าตั้งต้นตั้งตามคู่มือ server แล้ว
- บรรทัดที่เขียนว่า `CHECK` ควรเช็กกับบัญชีของเราก่อน

```bash
check_qos                     # ดูชื่อ QoS ที่บัญชีเราใช้ได้ ต้องมี gpu40g และ cpu40h (หรือชื่อที่ใกล้เคียง)
ldd --version | head -1       # ดูเวอร์ชัน glibc (CU-AIM = 2.17 ต่ำกว่า 2.28 จึงต้องใช้ container ตามค่าตั้งต้น)
sinfo -o "%20P %5a %10l %12G %N"   # ดูชื่อ partition และ gres ที่มีจริง
```

| ค่าใน server.env | ค่าตั้งต้น | ต้องแก้เมื่อไหร่ |
|---|---|---|
| `QOS` / `GPU_GRES` | `gpu40g` / `gpu:7g.40gb:1` (A100-40GB เต็มใบ) | `check_qos` ไม่มี `gpu40g` ให้ใส่ชื่อที่มี |
| `CPU_QOS` | `cpu40h` | `check_qos` ไม่มีชื่อนี้ |
| `CPU_PARTITION` | ว่าง (ใช้ partition default) | งาน analysis ถูก reject ให้ดูชื่อจาก `sinfo` |
| `ENV_KIND` | `container` (Singularity) | glibc ≥ 2.28 จะเปลี่ยนเป็น `conda` ก็ได้ ถ้าไม่แน่ใจให้คงค่าเดิม |
| `TORCH_INDEX` | `cu128` | ดูข้อถัดไป |
| `MAX_PARALLEL` | `2` | ให้รันพร้อมกันกี่โมเดล ถ้าคิวว่างเพิ่มได้ |
| `TIME_GPU` | `08:00:00` | ต้องไม่เกินคอลัมน์ `MaxWall` ของ QoS ที่ใช้ (`gpu40g` = 8 ชม., `gpu40g64` = 12 ชม., `gpu10gh` = 4 วัน แต่ slice เล็กแค่ 10 GB ต้องตั้ง `SEEDS_IN_PARALLEL=0`) |

**เช็กเวอร์ชัน CUDA driver (ไม่บังคับ):** ใช้ GPU slice เล็กสุดชั่วคราวผ่าน QoS interactive

```bash
srun -p gpu --qos=interactive-gpu --gres=gpu:1g.5gb:1 -c2 --mem=8G --pty nvidia-smi
```

- ดูช่อง "CUDA Version" มุมขวาบน
- 12.8 ขึ้นไป: ใช้ `cu128` ได้
- 13.x: เปลี่ยนเป็น `cu130` ได้ แต่ `cu128` ก็ใช้ได้เช่นกัน

**วิธีแก้ไฟล์:** `nano scripts/server/server.env` แล้วกด Ctrl+O เพื่อบันทึก และ Ctrl+X เพื่อออก

> ห้ามใส่ token หรือรหัสผ่านใน `server.env` เพราะไฟล์นี้อยู่ใน git

---

## 5. (ทางเลือก) อัปโหลดผลเดิมจาก Colab

ถ้าอยากให้ server ข้ามงานที่ Colab ทำเสร็จแล้ว ให้ copy ผลขึ้นไปจาก**เครื่องตัวเอง** โดยรันใน PowerShell:

```powershell
cd C:\Users\USER\Documents\bioinfo\gabm
# ผล feasibility: โมเดลที่มีอยู่ในไฟล์แล้วจะไม่รัน notebook 02 ซ้ำ
scp outputs/feasibility/feasibility.csv <username>@<ที่อยู่ CU-AIM>:/data/project/<กลุ่ม>/<โฟลเดอร์ของเรา>/gabm/outputs/feasibility/
# run ที่เสร็จแล้ว (ถ้าดาวน์โหลดมาจาก Google Drive): run ที่ config ตรงกันจะถูกข้าม
scp -r outputs/runs/* <username>@<ที่อยู่ CU-AIM>:/data/project/<กลุ่ม>/<โฟลเดอร์ของเรา>/gabm/outputs/runs/
```

- ถ้าโฟลเดอร์ `outputs/feasibility` ยังไม่มีบน server ให้รัน `mkdir -p outputs/feasibility` บน server ก่อน
- ใช้ Cyberduck หรือ WinSCP ลากวางแทนก็ได้ (คู่มือ server 1.4)
- ข้ามขั้นนี้ได้: server จะรันใหม่ทั้งหมด ใช้เวลานานขึ้นแต่ผลไม่ต่างกัน

---

## 6. ติดตั้ง environment (ครั้งเดียว)

```bash
bash -l scripts/server/setup_env.sh
```

สคริปต์จะทำสามอย่าง:
1. ดึง container `python:3.13-slim` มาเป็น `containers/python313.sif`
2. สร้าง venv ที่ `.venv-gabm/`
3. ติดตั้ง torch 2.11.0, transformers 5.20.1, bitsandbytes 0.50.2 และแพ็กเกจอื่น ซึ่งเป็นเวอร์ชันเดียวกับที่ Colab ใช้

ใช้เวลาประมาณ 10–20 นาที ถ้าสำเร็จ บรรทัดท้ายจะเป็น:

```
python ok | torch 2.11.0+cu128 | transformers 5.20.1 | bitsandbytes 0.50.2
Environment ready. Next: bash scripts/server/download.sh
```

ถ้าขึ้น `singularity: command not found` ให้ทำตามนี้:

```bash
module avail 2>&1 | grep -i -E "singularity|apptainer"
```

- ถ้าเจอ module ให้ใส่ใน `server.env` เป็น `MODULE_LOAD="module load <ชื่อที่เจอ>"`
- ถ้าเจอแต่ apptainer ให้ใส่ `SINGULARITY_BIN="apptainer"` ด้วย

จากนั้นรัน `setup_env.sh` ใหม่ ส่วนที่ทำเสร็จแล้วจะถูกข้าม

---

## 7. ใส่ HF token และดาวน์โหลดโมเดล

**ใส่ token** ทำครั้งเดียว ต้องใช้เฉพาะ llama8b

```bash
mkdir -p .hf_cache
nano .hf_cache/token          # วาง token (hf_...) บรรทัดเดียว บันทึกแล้วออก
chmod 600 .hf_cache/token     # ให้อ่านได้เฉพาะเรา
```

- `.hf_cache/` อยู่ใน `.gitignore` แล้ว token จึงไม่หลุดเข้า git
- ไม่ควรใช้ `echo "hf_..." > ...` เพราะ token จะไปค้างใน bash history

**ดาวน์โหลดโมเดล** 5 ตัว รวมประมาณ 80 GB

```bash
bash -l scripts/server/download.sh
```

- ต้องรันบน login node เพราะ compute node อาจออกอินเทอร์เน็ตไม่ได้
- งาน GPU ทั้งหมดอ่านโมเดลจาก `.hf_cache` แบบ offline
- ถ้าเน็ตหลุดกลางทาง ให้รันคำสั่งเดิมอีกครั้ง ไฟล์ที่โหลดเสร็จแล้วจะไม่โหลดซ้ำ

ถ้ากลัว ssh หลุดระหว่างดาวน์โหลด ให้รันแบบ background แล้วค่อยกลับมาดู log:

```bash
mkdir -p logs
nohup bash -l scripts/server/download.sh > logs/download.log 2>&1 &
tail -f logs/download.log     # Ctrl+C ออกจากการดู log (การดาวน์โหลดยังทำต่อ)
```

ผลลัพธ์ที่ควรได้คือ `ok` ครบ 5 บรรทัด
- ถ้า llama8b ขึ้น `FAIL ... gated` แปลว่ายังไม่ได้กด accept licence หรือ token ผิด
- แก้แล้วรันใหม่ได้เลย

---

## 8. Smoke test

ทดสอบสั้น ๆ ประมาณ 10 นาที ก่อนส่งงานใหญ่ โดยใช้โมเดลแรกใน `MODELS` ตรวจว่า:
- GPU ใช้ได้
- torch เห็น CUDA
- โมเดลอยู่ใน cache
- simulation รันได้ 2 วัน

```bash
bash scripts/server/submit_all.sh smoke
squeue -u $USER                          # ดูสถานะ: PD = รอคิว, R = กำลังรัน
tail -f logs/gabm-smoke_<jobid>.out      # jobid คือเลขที่คำสั่งแรกพิมพ์ออกมา
```

**ผ่าน** เมื่อบรรทัดท้ายเป็น `smoke test OK for llama8b ...`
- ระหว่างรันจะมีบรรทัดบอกว่าแต่ละวันใช้กี่วินาที เอาไปประมาณเวลาได้: วินาทีต่อวัน × 180 × 2 (C + D) ≈ เวลาต่อโมเดล
- run ของ smoke มี tag `smoke` ซึ่งไม่ถูกนับในการวิเคราะห์ใด ๆ

ถ้าไม่ผ่าน ให้ดูข้อความ error ท้าย log แล้วเทียบกับ [ข้อ 14](#14-แก้ปัญหาที่อาจเจอ)

---

## 9. รันจริง

```bash
bash scripts/server/submit_all.sh
```

คำสั่งนี้ส่งงานสองแบบ:

| งาน | ทรัพยากร | ทำอะไร |
|---|---|---|
| `gabm-model` (array 1 task ต่อ 1 โมเดล) | A100-40GB 1 ใบ, QoS `gpu40g` | check → feasibility (เฉพาะโมเดลที่ยังไม่อยู่ใน `feasibility.csv`) → C × 3 seeds → D × 3 seeds |
| `gabm-analyze` | CPU, QoS `cpu40h` | เริ่มเองหลัง array จบ แล้วรัน notebook 04 และ 05 (`RUN_LLM=False`) ทุกโมเดล |

- **3 seeds รันพร้อมกันบน GPU ใบเดียว** ใช้ VRAM รวมประมาณ 30 GB
- **D รันเฉพาะเมื่อ C ผ่าน** คือ C ครบทุก seed และ fallback < 5% ซึ่งเป็นเกณฑ์เดียวกับ notebook 05
- **บันทึก checkpoint ทุกวันจำลอง** ถ้างานหมดเวลา (`TIME_GPU` = 8 ชม. ตามเพดานของ QoS `gpu40g`) หรือถูก kill ให้รัน `bash scripts/server/submit_all.sh` ใหม่
  - run ที่เสร็จแล้วจะถูกข้าม และ run ที่ค้างจะทำต่อจากวันล่าสุด
- **feasibility พังไม่ทำให้งานหยุด** เพราะ C/D ไม่ได้ใช้ผล feasibility งานจะไปต่อเอง

---

## 10. ติดตามงาน / ยกเลิกงาน

```bash
squeue -u $USER                                  # งานของเราที่รอ/กำลังรัน
tail -f logs/gabm-model_<arrayid>_<task>.out     # task 0 = llama8b, 1 = typhoon8b, ... ตามลำดับใน MODELS
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,ExitCode   # งานที่จบแล้ว: COMPLETED / FAILED / TIMEOUT
ls outputs/runs/                                 # run ที่สร้างแล้ว
scancel -f <jobid>                               # ยกเลิกงาน (คู่มือ server 4.5)
```

ถ้าเลิกดู log ด้วย Ctrl+C หรือปิด ssh งานก็ยังรันต่อ เพราะงานอยู่ใน SLURM ไม่ได้ผูกกับ terminal

---

## 11. ดึงผลกลับมาที่เครื่อง

ไฟล์ผลลัพธ์บน server:

| ไฟล์ | คืออะไร |
|---|---|
| `outputs/runs/<run_id>/` | ผลแต่ละ run หน้าตาเหมือนบน Colab |
| `outputs/executed/*.ipynb` | notebook 02/04/05 ที่รันแล้ว มีตารางและกราฟครบ เปิดบนเครื่องหรือ Colab ได้เลย |
| `outputs/figures/` | กราฟ |
| `outputs/metrics.csv` | ตัวชี้วัดของทุก run ในไฟล์เดียว |
| `outputs/feasibility/feasibility.csv` | ผล feasibility |
| `logs/` | log ของแต่ละงาน |

ดึงกลับด้วยคำสั่งใน PowerShell บนเครื่องตัวเอง:

```powershell
cd C:\Users\USER\Documents\bioinfo\gabm
scp -r <username>@<ที่อยู่ CU-AIM>:/data/project/<กลุ่ม>/<โฟลเดอร์ของเรา>/gabm/outputs .
```

หรือใช้ Cyberduck / WinSCP ก็ได้

**ถ้าจะวิเคราะห์ต่อบน Colab:** อัปโหลด `outputs/runs/` ขึ้น Drive ในโฟลเดอร์เดิม
- ผลใหม่ไม่ปนกับผลเก่า เพราะ notebook กรองด้วย `DAYS` และเลือก run ล่าสุดเมื่อซ้ำกัน

---

## 12. อัปเดตโค้ดรอบถัดไป (git pull)

เมื่อแก้โค้ดบนเครื่องและ push ขึ้น GitHub แล้ว ให้ดึงโค้ดใหม่บน server:

```bash
cd /data/project/<กลุ่ม>/<โฟลเดอร์ของเรา>/gabm
git stash            # เก็บสิ่งที่แก้บน server ไว้ก่อน (เช่น server.env); outputs/ ไม่อยู่ใน git จึงไม่โดนแตะ
git pull
git stash pop        # เอาที่แก้ไว้กลับมา
```

- ถ้าไม่ได้แก้ไฟล์ใดใน git เลย `git stash` จะบอกว่า "No local changes to save" และ `git stash pop` จะ error ได้ ไม่ต้องสนใจ
- ถ้า `git stash pop` แจ้ง conflict ที่ `server.env`:
  1. เปิดไฟล์ หาช่วงระหว่าง `<<<<<<<` กับ `>>>>>>>` ให้เหลือค่าที่ต้องการ แล้วลบบรรทัดเครื่องหมายพวกนั้นออก
  2. รัน `git reset && git stash drop`
- **ไม่ต้องติดตั้ง environment หรือดาวน์โหลดโมเดลใหม่** ยกเว้น `requirements-server.txt` เปลี่ยน (ให้รัน `setup_env.sh` อีกครั้ง) หรือเพิ่มโมเดลใหม่ (ให้รัน `download.sh`)
- **ถ้าแก้ `config/base.yaml`** (พารามิเตอร์ simulation หรือรายชื่อโมเดล) run_id จะเปลี่ยน
  - งานรอบถัดไปจะรันใหม่ทั้งหมด ไม่ทับผลเก่า
  - เป็นพฤติกรรมที่ตั้งใจไว้ เพื่อไม่ให้ผลจาก config ต่างกันปนกัน

---

## 13. ปรับแต่งการรัน

แก้ใน `scripts/server/server.env` แล้วรัน `submit_all.sh` ใหม่

| อยากทำ | แก้ |
|---|---|
| รันบางโมเดล | `MODELS="typhoon8b qwen7b"` |
| เพิ่ม mistral7b (negative control) | เพิ่มใน `MODELS` แล้วรัน `download.sh` อีกครั้ง |
| ทำ robustness ของ notebook 04 | `STAGES="feasibility C D order sens"` (`order` = สลับลำดับ action, `sens` = sensitivity sweeps) |
| ได้คิวเร็วขึ้นด้วย GPU ที่เล็กลง | `GPU_GRES="gpu:3g.20gb:1"` + `QOS` ที่คู่กัน (ดูจาก `check_qos`) + `SEEDS_IN_PARALLEL=0` (seeds รันทีละตัว) |
| รันพร้อมกันหลายโมเดลขึ้น | `MAX_PARALLEL=3` (ควรเกรงใจคนอื่นในแล็บ) |
| รันเฉพาะ analysis ใหม่ | `bash scripts/server/submit_all.sh analyze` |

ถ้าเปลี่ยน `DAYS` ต้องเปลี่ยน `DAYS` ใน notebook 04/05 ให้ตรงกันด้วย ไม่อย่างนั้น notebook จะหาผลไม่เจอ

**รันขั้นเดียวด้วยมือ** (debug): ให้ขอ GPU แบบ interactive ก่อน แล้วรันผ่าน container

```bash
srun -p gpu --qos=gpu40g --gres=gpu:7g.40gb:1 -c8 --mem=64G --time=02:00:00 --pty bash -l
# พอได้ shell บน GPU node แล้ว:
run() { bash -lc 'source scripts/server/common.sh && activate_env && in_env python scripts/run_server.py "$@"' _ "$@"; }
run --model qwen7b --stage C --seeds 1
run --help                                       # ดูตัวเลือกทั้งหมด
exit                                             # คืน GPU เมื่อเสร็จ
```

ฟังก์ชัน `run` เปิด bash ใหม่ทุกครั้ง เพราะ `common.sh` ตั้ง `set -e` ไว้
ถ้า `source` ตรง ๆ ใน shell ที่ใช้อยู่ คำสั่งไหนพังเพียงครั้งเดียว session จะหลุดทันที

---

## 14. แก้ปัญหาที่อาจเจอ

| อาการ (ใน log) | สาเหตุ / วิธีแก้ |
|---|---|
| `singularity: command not found` | ดู [ข้อ 6](#6-ติดตั้ง-environment-ครั้งเดียว): ตั้ง `MODULE_LOAD` หรือ `SINGULARITY_BIN="apptainer"` |
| `CUDA is not available` | ตรวจว่าขอ GPU ถูก (`GPU_GRES`, `QOS`) หรือ driver เก่ากว่า `TORCH_INDEX` ให้ลอง `cu126` แล้วรัน `setup_env.sh` ใหม่ |
| `... is not in the local cache ... HF_HUB_OFFLINE=1` | ยังไม่ได้ดาวน์โหลดโมเดลนั้น ให้รัน `download.sh` |
| `FAIL llama8b ... gated` | ยังไม่ accept licence บน huggingface.co หรือ `.hf_cache/token` ผิด |
| `sbatch: error: Invalid qos specification` | ชื่อ QoS ไม่ตรงกับบัญชี ให้ดู `check_qos` แล้วแก้ `QOS` / `CPU_QOS` |
| `sbatch: error: ... partition` (งาน analysis) | ตั้ง `CPU_PARTITION` ตามชื่อใน `sinfo` |
| `CUDA out of memory` | ตั้ง `SEEDS_IN_PARALLEL=0` ให้ seeds รันทีละตัว |
| `D gate not met ... fallback rate` | โมเดลนี้ตอบผิด schema บ่อยเกินไป (fallback ≥ 5%) จึงไม่รัน D (ตั้งใจ ตามเกณฑ์ของ notebook 05) |
| `D gate not met ... missing for seeds` | C ยังไม่ครบ ให้ดู log ของ stage C ว่าพังที่ไหน |
| งานจบด้วย `TIMEOUT` | รัน `submit_all.sh` อีกครั้ง งานจะทำต่อจาก checkpoint |
| ตาราง VRAM ใน `feasibility.csv` ว่าง | ปกติของ GPU แบบ MIG (`nvidia-smi` ไม่รายงานหน่วยความจำราย slice) ตัวเลขอื่นใช้ได้ตามปกติ |
| analysis รันก่อนที่โมเดลที่ resubmit จะเสร็จ | รัน `bash scripts/server/submit_all.sh analyze` หลังโมเดลเสร็จ |

---

## 15. ไฟล์ใน scripts/server ทำอะไร

| ไฟล์ | หน้าที่ |
|---|---|
| `server.env` | ค่าตั้งทั้งหมด (path, environment, SLURM, รายชื่อโมเดล, DAYS, stages) |
| `common.sh` | ฟังก์ชันกลาง: โหลด `server.env`, เปิด environment, `in_env` (รันคำสั่งใน container) |
| `setup_env.sh` | ติดตั้ง environment ครั้งเดียว |
| `download.sh` | ดาวน์โหลดโมเดลลง `.hf_cache` (เรียก `scripts/download_models.py`) |
| `submit_all.sh` | ส่งงาน: `smoke` / ไม่ใส่อะไร (รันทั้งหมด) / `analyze` |
| `smoke.sbatch` | งานทดสอบ 2 วัน |
| `run_model.sbatch` | งาน GPU ต่อ 1 โมเดล (ห้าม `sbatch` ตรง ให้ส่งผ่าน `submit_all.sh`) |
| `analyze.sbatch` | รัน analysis ของ notebook 04/05 ด้วย papermill |
| `../run_server.py` | ตัวรันจริง (ใช้ได้ทั้งจาก sbatch และรันด้วยมือ) |

รายละเอียดภาษาอังกฤษอยู่ที่ [scripts/server/README.md](scripts/server/README.md)
