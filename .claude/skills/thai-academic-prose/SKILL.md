---
name: thai-academic-prose
description: House style for Thai academic writing in report/ — papers, progress reports, and any prose an advisor or reviewer reads. Use when drafting or revising anything in report/, or when prose reads as robotic, note-like, or over-emphasised. Converts engineering-memo register into explanatory academic Thai.
---

# Thai academic prose

Everything in `report/` is read by an advisor or a reviewer, not by the person who
ran the jobs. The failure mode is prose that still sounds like the working notes it
came from: terse, shouted in bold, addressed to whoever is holding the keyboard.
The target register is **explanatory** — a knowledgeable colleague walking a reader
through what was done and what it means.

## The six rules

### 1. Bold is structure, not emphasis
At most one bold span per section, and only to introduce a term the reader will meet
again. Never bold a number, a finding, or a whole clause. If a sentence needs bold to
land, rewrite the sentence.

- ✗ `ผลของการจูนคือ **−0.1814** และ **ต้องระบุให้ชัดว่า** นี่คือค่าต่ำสุด`
- ✓ `ผลของการจูนในตระกูล Llama อยู่ที่ −0.1814 ค่า p ที่ได้คือค่าต่ำสุดที่การออกแบบนี้ให้ได้`

### 2. Describe, never instruct the reader
Drop `ต้องระบุว่า`, `ต้องอ่านคู่กับ`, `ต้องระวังว่า`, `ข้อที่ต้องย้ำคือ`. These address a
collaborator. A paper states the fact and lets the reader draw the caution.

- ✗ `ต้องระบุให้ชัดว่า SD ที่ต่ำนี้มาจากประชากรขนาดใหญ่`
- ✓ `SD ที่ต่ำข้าม seed มาจากประชากร N = 1,000,000 ซึ่งกลบความผันผวนเชิงสุ่มของ SEIR`

### 3. One claim per sentence
Thai academic sentences run roughly 15–35 words. A sentence carrying four findings
separated by spaces is a list that lost its bullets. Split it, or make it a table.

### 4. No notes punctuation in prose
`·` belongs in `working_notes_TH.md`. In prose use a clause, a comma, or a table.
Limit `—` to one per page; prefer parentheses or a new sentence.

### 5. Lead with the finding, not a hook
Delete teasers: `ข้อค้นพบที่น่าสนใจคือ`, `ที่สำคัญกว่านั้น`, `สิ่งที่ต้องรายงานอย่างตรงไปตรงมาคือ`.
Open the paragraph with the result; the reader decides whether it is interesting.

- ✗ `ข้อสังเกตที่ต้องรายงานอย่างตรงไปตรงมาคือ สี่ในห้าโมเดลให้ q̄_out มากกว่า 1`
- ✓ `สี่ในห้าโมเดลให้ q̄_out มากกว่า 1 ซึ่งหมายความว่า agent ออกนอกบ้านมากกว่าวันปกติ`

### 6. Hedge by scope, not by exhortation
Keep every caveat — honest limits are this project's strength — but phrase them as the
boundary of the claim rather than as a warning.

- ✗ `ตัวเลขนี้ไม่ควรอ่านเป็นความมั่นใจ`
- ✓ `ข้อสรุปนี้ครอบคลุมเฉพาะการสุ่มบุคลิกชุดเดียวที่ใช้ในทุก seed`

## Paragraph shape

Three to six sentences. The first states the claim, the middle gives the evidence with
its numbers, the last gives the scope or the link to the next section. A paragraph that
is one sentence long is a caption; a paragraph over eight sentences is two paragraphs.

## Numbers

Prose carries the precision that matters (`0.638`, `ราว 46 เท่า`); tables carry full
precision (`0.6381 (0.0040)`). Never repeat a table's digits in the sentence beside it.
Every number must already exist in a verified source — never compute one while writing.

## Terminology

English technical terms stay in English and stay consistent: `attack rate`, `seed`,
`prompt`, `fallback`, `persona`. Do not translate a term once and transliterate it later.
Model keys, file paths and run ids are code, in backticks.

## Revising existing text

Work section by section. For each one: count bold spans and cut to at most one; find
every sentence that addresses the reader and restate it as fact; split sentences past
~35 words; replace `·` runs with clauses or a table. Do not change any number, any
caveat, or any claim — this is a register change, not a content change. If a rewrite
would alter what is being claimed, leave it and flag it.
