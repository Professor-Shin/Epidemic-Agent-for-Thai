"""LLM backends, the resident prompt, and output validation (guideline section 7.3).

Validation policy: parse JSON -> action must be in the allowed list -> otherwise retry once with a reminder ->
otherwise keep the agent's previous action (or its default on day 0). Fallbacks are never silent: every record
keeps the raw output, the retry output, and flags valid / retried / fallback.
"""
import gc
import json
import random
import re
import time

RESIDENT_TEMPLATE = """You are simulating the daily choice of one resident in a synthetic epidemic scenario set in a province in Thailand. A new respiratory infection is spreading. Decide as this person would, given their situation.

Resident profile:
{persona}

Today is day {day} of the simulation.
- Your health: {health}
- Newly reported cases in your province today: {today_per100k} per 100,000 people (7-day average {avg7_per100k}; trend: {trend})
- Current government measures: {policy}

Choose exactly ONE main activity for today:
{action_list}

Reply with JSON only, no other text:
{{"action": "<one of: {action_names}>", "reason": "<at most 15 words>"}}"""

RETRY_SUFFIX = "\n\nYour previous reply was not valid. Reply with one JSON object only, and the action must be exactly one of: {action_names}."

JSON_RE = re.compile(r"\{.*?\}", re.S)


def build_resident_prompt(persona, day, health, report, policy_text, actions):
    action_list = "\n".join(f"- {a}: {desc}" for a, desc in actions.items())
    return RESIDENT_TEMPLATE.format(persona=persona, day=day, health=health, policy=policy_text,
                                    action_list=action_list, action_names=", ".join(actions), **report)


def parse_choice(text, allowed, key="action"):
    """Return (choice, reason, error). choice is None when the output is invalid."""
    m = JSON_RE.search(text or "")
    if not m:
        return None, None, "no_json"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, None, "bad_json"
    choice = str(obj.get(key, "")).strip().lower()
    if choice not in allowed:
        return None, obj.get("reason"), f"not_allowed:{choice[:30]}"
    return choice, obj.get("reason"), None


# ---------------------------------------------------------------- backends

class MockBackend:
    """No-GPU stand-in for testing the pipeline. Reacts to symptoms and case level; ~5% malformed replies."""
    name = "mock"

    def __init__(self, invalid_rate=0.05):
        self.invalid_rate = invalid_rate

    def generate(self, prompts, temperature=0.7, seed=0, max_new_tokens=64):
        out = []
        for j, p in enumerate(prompts):
            rng = random.Random(f"{seed}-{j}-{p[-400:]}")
            if rng.random() < self.invalid_rate:
                out.append("I think I will probably stay home today.")
                continue
            if '"level"' in p:   # leader prompt
                cases = float(re.search(r"7-day average of newly reported cases: ([\d.]+)", p).group(1))
                level = "strict" if cases > 60 else "limited" if cases > 20 else "open"
                out.append(json.dumps({"level": level, "reason": "mock rule"}))
                continue
            cases = float(re.search(r"7-day average ([\d.]+)", p).group(1))
            sick = "fever and cough" in p
            p_home = min(0.95, 0.15 + cases / 150 + (0.6 if sick else 0))
            if rng.random() < p_home:
                a = "stay_home"
            else:
                a = rng.choice(["work_onsite", "work_onsite", "essential_errand", "visit_family"])
            if a not in p.split("Choose exactly ONE")[-1]:
                a = "essential_errand"
            out.append(json.dumps({"action": a, "reason": "mock decision"}))
        return out

    def unload(self):
        pass


class TransformersBackend:
    """Hugging Face transformers with 4-bit bitsandbytes quantisation. Works on a free Colab T4."""

    def __init__(self, model_id, batch_size=40, max_prompt_tokens=1024, load_in_4bit=True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch = torch
        self.name = model_id
        self.batch_size = batch_size
        self.max_prompt_tokens = max_prompt_tokens
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16) if load_in_4bit else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=quant, device_map="auto", torch_dtype=torch.float16)
        self.model.eval()

    def _chat(self, prompt):
        # single user turn: some chat templates (Gemma, Mistral) reject a system role
        return self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                            tokenize=False, add_generation_prompt=True)

    def generate(self, prompts, temperature=0.7, seed=0, max_new_tokens=64, top_p=0.9):
        torch = self.torch
        out = []
        for b in range(0, len(prompts), self.batch_size):
            batch = [self._chat(p) for p in prompts[b:b + self.batch_size]]
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_prompt_tokens, add_special_tokens=False).to(self.model.device)
            torch.manual_seed(seed + b)
            kw = dict(do_sample=True, temperature=temperature, top_p=top_p) if temperature > 0 else dict(do_sample=False)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id, **kw)
            out += self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return out

    def unload(self):
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()


class VLLMBackend:
    """Optional faster backend (AWQ checkpoints, fp16 on T4). If it fails to install or load, use transformers."""

    def __init__(self, model_id, max_model_len=2048, gpu_memory_utilization=0.85):
        from vllm import LLM
        self.name = model_id
        self.llm = LLM(model=model_id, dtype="half", quantization="awq" if "awq" in model_id.lower() else None,
                       max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization)

    def generate(self, prompts, temperature=0.7, seed=0, max_new_tokens=64, top_p=0.9):
        from vllm import SamplingParams
        params = [SamplingParams(temperature=temperature, top_p=top_p if temperature > 0 else 1.0,
                                 max_tokens=max_new_tokens, seed=seed + j) for j in range(len(prompts))]
        convs = [[{"role": "user", "content": p}] for p in prompts]
        res = self.llm.chat(convs, params, use_tqdm=False)
        return [r.outputs[0].text for r in res]

    def unload(self):
        import torch
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()


def load_backend(llm_cfg, model_key=None, backend=None):
    backend = backend or llm_cfg["backend"]
    if backend == "mock":
        return MockBackend()
    m = llm_cfg["models"][model_key]
    if backend == "vllm":
        if not m.get("vllm_id"):
            raise ValueError(f"{model_key} has no vllm_id; use backend='transformers'")
        return VLLMBackend(m["vllm_id"])
    return TransformersBackend(m["hf_id"], batch_size=llm_cfg["batch_size"], max_prompt_tokens=llm_cfg["max_prompt_tokens"])


# ---------------------------------------------------------------- deciders

def decide_batch(backend, prompts, allowed, fallbacks, llm_cfg, seed, key="action"):
    """Generate, validate, retry invalid replies once, then fall back. Returns one record per prompt."""
    kw = dict(temperature=llm_cfg["temperature"], max_new_tokens=llm_cfg["max_new_tokens"])
    t0 = time.time()
    raw = backend.generate(prompts, seed=seed, **kw)
    records = []
    for r in raw:
        choice, reason, err = parse_choice(r, allowed, key)
        records.append({"raw": r, "choice": choice, "reason": reason, "error": err, "retried": False, "raw_retry": None})
    bad = [j for j, rec in enumerate(records) if rec["choice"] is None]
    for _ in range(llm_cfg.get("retry_invalid", 1)):
        if not bad:
            break
        suffix = RETRY_SUFFIX.format(action_names=", ".join(allowed))
        raw2 = backend.generate([prompts[j] + suffix for j in bad], seed=seed + 100003, **kw)
        for j, r in zip(bad, raw2):
            choice, reason, err = parse_choice(r, allowed, key)
            records[j].update(retried=True, raw_retry=r, error_retry=err)
            if choice is not None:
                records[j].update(choice=choice, reason=reason)
        bad = [j for j in bad if records[j]["choice"] is None]
    for j, rec in enumerate(records):
        rec["valid"] = rec["choice"] is not None
        rec["fallback"] = not rec["valid"]
        if rec["fallback"]:
            rec["choice"] = fallbacks[j]
    elapsed = time.time() - t0
    return records, elapsed
