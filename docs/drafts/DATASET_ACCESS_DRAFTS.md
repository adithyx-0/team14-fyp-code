# Dataset access request drafts (DRAFTS ONLY, nothing sent)

`instructions.md` (sections 3 and 7) says both datasets are "not acquired" and
names no source, contact or link, and `data/secondary/` and `data/reference/`
are empty. Everything in [square brackets] must be filled in by you. Items
marked VERIFY are things I believe but could not check from the project files
(no web access used for this).

Who should send these: whoever the dataset owner will recognise, ideally from
the university email, with the guide copied. Check the guide's name first:
the project notes say "Dr. Veena G" but your Telegram chat shows
"Dr.VeenaDileep". I left a placeholder.

---

## 1. Tamil + Malayalam misogyny meme dataset

What we need and why (for the request and for the guide):
- Secondary dataset for cross-lingual transfer in the project plan: train on
  one language, test on the other, and compare against our Malayalam-only
  political-meme results.
- Research use only. We will not redistribute the data.

Where it likely comes from (VERIFY): the Tamil and Malayalam misogyny meme
shared task run with the DravidianLangTech workshop series (the same workshop
family as the SYNAPSE paper). Confirm the exact task name, year, organisers
and the access form on the workshop or shared-task page, or on the paper that
introduces the dataset, before sending.

Registration steps to check (VERIFY, shared tasks usually work like this):
1. Find the shared-task page and its dataset link (Codabench, Google Form or
   a GitHub release).
2. Register the team and accept the data-use terms.
3. Some releases give train and dev only, and keep test labels hidden;
   confirm which splits and labels we will get, because cross-lingual
   evaluation needs labelled test data.
4. Cite the dataset paper in the report.

**Draft email**

Subject: Request for access to the Tamil and Malayalam misogyny meme dataset ([task/year])

Dear [organiser name / Sir or Madam],

We are final-year students of [department], [university], working on a
project titled "LLM-Augmented Multimodal Meme Understanding: Stance
Classification, Explainability, and Cross-Lingual Analysis for Low-Resource
Indian Languages" under the guidance of [Dr. guide name].

Our primary dataset is the DravidianLangTech 2026 Malayalam political meme
set. For the cross-lingual part of the project we would like to use the
[task name, year] Tamil and Malayalam misogyny meme dataset, to study how a
model fine-tuned in one language transfers to the other.

Could you please let us know how to obtain the dataset (train, development
and, if possible, labelled test data) and whether there is a registration
form or data-use agreement we should complete? We will use the data only for
academic research, will not redistribute it, and will cite [dataset paper].

Our guide, [Dr. guide name], is copied on this email.

Thank you for your time.

Regards,
[Name], Team 14, [course code 22AIE498], [university]
[contact email]

---

## 2. Hateful Memes (English reference set)

What we need and why:
- A small English reference point for the zero-shot prompting arm only, to
  put our Malayalam numbers in context. Not a full benchmark. A subsample
  (for example 100 to 200 images) is enough, which matters because of
  our Mac-only compute.

Where it comes from (VERIFY): Facebook AI's Hateful Memes Challenge dataset
(Kiela et al., NeurIPS 2020), distributed through a DrivenData / Meta page
that requires registering and accepting a licence before download. Confirm the
current download page and licence terms before you register.

Registration steps to check (VERIFY):
1. Create an account on the hosting platform with the university email.
2. Read and accept the dataset licence (it restricts use and redistribution;
   do not put the images in a public repository).
3. Download the data, using the labelled splits (the dev or `dev_seen` style
   files carry labels; some test files do not).
4. Store it under `data/reference/` and keep it out of git (large and
   licensed).

This one is a self-service registration, so you may not need an email. If the
platform has a contact route for access problems, use this draft.

**Draft email**

Subject: Hateful Memes dataset access for an academic project

Dear Hateful Memes dataset team,

We are final-year students at [university] working on multilingual meme
understanding for low-resource Indian languages ([course code], guide:
[Dr. guide name]). We would like to use the Hateful Memes dataset
(Kiela et al., 2020) as an English reference point for zero-shot evaluation
of vision-language models, using a small labelled subsample.

We have [registered / are unable to complete registration because ...].
Could you please advise how to obtain the labelled data? We will follow the
dataset licence, use it for academic research only, and not redistribute it.

Regards,
[Name], Team 14, [university], [contact email]

---

## Before sending
- [ ] Fill every [bracket]; confirm the guide's name.
- [ ] Verify each VERIFY item on the real pages, and fix the dataset names.
- [ ] Ask the guide to approve the wording, and whether she wants to be copied.
- [ ] Log the request date in `HANDOFF.md`; approval lag is the unknown.
