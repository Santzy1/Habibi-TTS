import random
from importlib.resources import files

import editdistance
import regex as re
import torch
import torchaudio
from f5_tts.model.modules import MelSpec
from tqdm import tqdm

from habibi_tts.model.utils import dialect_id_map, text_list_formatter


def get_single_prompt(gen_dialect):
    if gen_dialect in ["ALG", "IRQ", "UAE"]:
        ref_audio = str(files("habibi_tts").joinpath(f"assets/{gen_dialect}.wav"))
    elif gen_dialect in ["EGY", "MAR", "MSA"]:
        ref_audio = str(files("habibi_tts").joinpath(f"assets/{gen_dialect}.mp3"))
    elif gen_dialect[:3] == "SAU":
        ref_audio = str(files("habibi_tts").joinpath(f"assets/{gen_dialect[4:]}.wav"))
    else:
        ref_audio = str(files("habibi_tts").joinpath("assets/MSA.mp3"))

    if gen_dialect == "MSA":
        ref_text = "كان اللعيب حاضرًا في العديد من الأنشطة والفعاليات المرتبطة بكأس العالم، مما سمح للجماهير بالتفاعل معه والتقاط الصور التذكارية."
    elif gen_dialect[:3] == "SAU":
        if gen_dialect[4:] == "Najdi":
            ref_text = "تكفى طمني انا اليوم ماني بنايم ولا هو بداخل عيني النوم الين اتطمن عليه."
        elif gen_dialect[4:] == "Hijazi":
            ref_text = "ابغاك تحقق معاه بس بشكل ودي لانه سلطان يمر بظروف صعبة شوية."
        elif gen_dialect[4:] == "Gulf":
            ref_text = "وين تو الناس متى تصحى ومتى تفطر وتغير يبيلك ساعة يعني بالله تروح الشغل الساعة عشره."
        else:
            raise ValueError(f"[Code utils_eval.py] unexpected gen_dialect: {gen_dialect}")
    elif gen_dialect == "UAE":
        ref_text = "قمنا نشتريها بشكل متكرر أو لما نلقى ستايل يعجبنا وحياناً هذا الستايل ما نحبه."
    elif gen_dialect == "ALG":
        ref_text = "أنيا هكا باغية ناكل هكا أني ن نشوف فيها الحاجة هذيكا."
    elif gen_dialect == "IRQ":
        ref_text = "يعني ااا ما نقدر ناخذ وقت أكثر، ااا لأنه شروط كلش يحتاجلها وقت."
    elif gen_dialect == "EGY":
        ref_text = "ايه الكلام. بقولك ايه. استخدم صوتي في المحادثات. استخدمه هيعجبك اوي."
    elif gen_dialect == "MAR":
        ref_text = "إذا بغيتي شي صوت باللهجة المغربية للإعلانات ديالك هذا أحسن واحد غادي تلقاه."
    else:
        ref_text = "كان اللعيب حاضرًا في العديد من الأنشطة والفعاليات المرتبطة بكأس العالم، مما سمح للجماهير بالتفاعل معه والتقاط الصور التذكارية."

    return ref_audio, ref_text


def get_inference_prompt(
    metainfo,
    mel_spec_kwargs,
    speed=1.0,
    target_rms=0.1,
    target_sample_rate=24000,
    wrap_text_with_dialect_id=False,
    single=False,
):
    mel_spectrogram = MelSpec(**mel_spec_kwargs)

    prompts_all = []
    for mi in tqdm(metainfo, desc="Processing prompts..."):
        ref_audio, ref_text, gen_text, gen_path, gen_dialect = mi

        # preprocess ref_audio to ref_mel
        if not single:
            ref_sr = ref_audio["sampling_rate"]
            ref_audio = torch.from_numpy(ref_audio["array"]).float().unsqueeze(0)
        else:  # single prompt case, i.o.t. compare with 11labs v3-a
            ref_audio, ref_text = get_single_prompt(gen_dialect)
            ref_audio, ref_sr = torchaudio.load(ref_audio)
        ref_rms = torch.sqrt(torch.mean(torch.square(ref_audio)))
        if ref_rms < target_rms:
            ref_audio = ref_audio * target_rms / ref_rms
        if ref_sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(ref_sr, target_sample_rate)
            ref_audio = resampler(ref_audio)
        ref_mel = mel_spectrogram(ref_audio)
        ref_mel = ref_mel.permute(0, 2, 1)

        # process text
        if len(ref_text[-1].encode("utf-8")) != 1:  # if no period
            gen_text = " " + gen_text
        else:
            ref_text = ref_text + " "
        final_text_list = [ref_text + gen_text]
        if wrap_text_with_dialect_id:
            dialect_id = dialect_id_map[gen_dialect[:3]]
            final_text_list = text_list_formatter(final_text_list, dialect_id=dialect_id)

        # cal duration in mel frame length
        ref_mel_len = ref_mel.shape[1]
        ref_text_len = len(ref_text.encode("utf-8"))
        gen_text_len = len(gen_text.encode("utf-8"))
        total_mel_len = ref_mel_len + int(ref_mel_len / ref_text_len * gen_text_len / speed)

        # add to prompt list
        prompts_all.append(
            (
                gen_path,
                ref_rms,
                ref_mel,
                ref_mel_len,
                total_mel_len,
                final_text_list,
            )
        )

    random.seed(666)
    random.shuffle(prompts_all)

    return prompts_all


def normalize_arabic_text(text):
    # Remove punctuation
    text = re.sub(r"[\p{p}\p{s}]", "", text)

    # Remove diacritics
    diacritics = r"[\u064B-\u0652]"  # Arabic diacritical marks (Fatha, Damma, etc.)
    text = re.sub(diacritics, "", text)

    # Normalize Hamzas and Maddas
    text = re.sub("پ", "ب", text)
    text = re.sub("ڤ", "ف", text)
    text = re.sub(r"[آ]", "ا", text)
    text = re.sub(r"[أإ]", "ا", text)
    text = re.sub(r"[ؤ]", "و", text)
    text = re.sub(r"[ئ]", "ي", text)
    text = re.sub(r"[ء]", "", text)
    text = re.sub(r"[ة]", "ه", text)

    # Transliterate Eastern Arabic numerals to Western Arabic numerals
    eastern_to_western_numerals = {
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
    }
    for eastern, western in eastern_to_western_numerals.items():
        text = text.replace(eastern, western)

    # Remove tatweel (kashida, u+0640)
    text = re.sub(r"\u0640", "", text)

    # Remove hmm-uhm-like words
    text = re.sub(r"اا+", "", text)

    # Normalize multiple whitespace characters into a single space
    text = re.sub(r"\s\s+", " ", text)

    return text.strip()


def word_error_rate(hypotheses, references, use_cer=False):
    scores = 0
    words = 0
    assert len(hypotheses) == len(references)
    for h, r in zip(hypotheses, references):
        if use_cer:
            h_list = list(h)
            r_list = list(r)
        else:
            h_list = h.split()
            r_list = r.split()
        words += len(r_list)
        scores += editdistance.eval(h_list, r_list)
    if words != 0:
        wer = 1.0 * scores / words
    else:
        wer = float("inf")
    return wer
