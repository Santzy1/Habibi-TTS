import json
import os
import re
import tempfile
from collections import OrderedDict
from functools import lru_cache
from importlib.resources import files

import click
import gradio as gr
import numpy as np
import psutil
import soundfile as sf
import torch
import torchaudio
from cached_path import cached_path
from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    tempfile_kwargs,
)
from f5_tts.model import DiT

from habibi_tts.infer.utils_infer import infer_process
from habibi_tts.model.utils import dialect_id_map


try:
    import spaces

    USING_SPACES = True
except ImportError:
    USING_SPACES = False


def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func


vocoder = load_vocoder()


# if not enough gpu memory, offload ckpt when switched
if USING_SPACES:
    lazy_load_model = False
else:
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
    elif torch.backends.mps.is_available():
        gpu_memory = psutil.virtual_memory().available / (1024**3)
    elif torch.xpu.is_available():
        gpu_memory = torch.xpu.get_device_properties(0).total_memory / (1024**3)
    else:
        gpu_memory = 0

    if gpu_memory > 12:
        lazy_load_model = False
    else:
        lazy_load_model = True


# common used configs and vocabs
v1_base_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
uni_vocab_path = str(cached_path("hf://SWivid/Habibi-TTS/Unified/vocab.txt"))
alg_vocab_path = str(cached_path("hf://SWivid/Habibi-TTS/Specialized/ALG/vocab.txt"))
egy_vocab_path = str(cached_path("hf://SWivid/Habibi-TTS/Specialized/EGY/vocab.txt"))
irq_vocab_path = str(cached_path("hf://SWivid/Habibi-TTS/Specialized/IRQ/vocab.txt"))
mar_vocab_path = str(cached_path("hf://SWivid/Habibi-TTS/Specialized/MAR/vocab.txt"))
msa_vocab_path = str(cached_path("hf://SWivid/Habibi-TTS/Specialized/MSA/vocab.txt"))
sau_vocab_path = str(cached_path("hf://SWivid/Habibi-TTS/Specialized/SAU/vocab.txt"))
uae_vocab_path = str(cached_path("hf://SWivid/Habibi-TTS/Specialized/UAE/vocab.txt"))

# text-to-speech language model gallery
uni_model_path = str(cached_path("hf://SWivid/Habibi-TTS/Unified/model_200000.safetensors"))
alg_model_path = str(cached_path("hf://SWivid/Habibi-TTS/Specialized/ALG/model_100000.safetensors"))
egy_model_path = str(cached_path("hf://SWivid/Habibi-TTS/Specialized/EGY/model_100000.safetensors"))
irq_model_path = str(cached_path("hf://SWivid/Habibi-TTS/Specialized/IRQ/model_100000.safetensors"))
mar_model_path = str(cached_path("hf://SWivid/Habibi-TTS/Specialized/MAR/model_100000.safetensors"))
msa_model_path = str(cached_path("hf://SWivid/Habibi-TTS/Specialized/MSA/model_200000.safetensors"))
sau_model_path = str(cached_path("hf://SWivid/Habibi-TTS/Specialized/SAU/model_200000.safetensors"))
uae_model_path = str(cached_path("hf://SWivid/Habibi-TTS/Specialized/UAE/model_100000.safetensors"))

unified_model = (
    load_model(
        DiT,
        v1_base_cfg,
        uni_model_path,
        vocab_file=uni_vocab_path,
    )
    if not lazy_load_model
    else [uni_model_path, uni_vocab_path]
)

tts_lang_model_collections = {
    "MSA (Modern Standard Arabic)": {
        "Unified": unified_model,
        "Specialized": [msa_model_path, msa_vocab_path]
        if lazy_load_model
        else load_model(DiT, v1_base_cfg, msa_model_path, vocab_file=msa_vocab_path),
    },
    "SAU (Najdi, Hijazi, Gulf, etc.)": {
        "Unified": unified_model,
        "Specialized": [sau_model_path, sau_vocab_path]
        if lazy_load_model
        else load_model(DiT, v1_base_cfg, sau_model_path, vocab_file=sau_vocab_path),
    },
    "UAE (Emirati)": {
        "Unified": unified_model,
        "Specialized": [uae_model_path, uae_vocab_path]
        if lazy_load_model
        else load_model(DiT, v1_base_cfg, uae_model_path, vocab_file=uae_vocab_path),
    },
    "ALG (Algerian & Algerian Saharan)": {
        "Unified": unified_model,
        "Specialized": [alg_model_path, alg_vocab_path]
        if lazy_load_model
        else load_model(DiT, v1_base_cfg, alg_model_path, vocab_file=alg_vocab_path),
    },
    "IRQ (Mesopotamian & North Mesopotamian)": {
        "Unified": unified_model,
        "Specialized": [irq_model_path, irq_vocab_path]
        if lazy_load_model
        else load_model(DiT, v1_base_cfg, irq_model_path, vocab_file=irq_vocab_path),
    },
    "EGY (Egyptian, Saidi, etc.)": {
        "Unified": unified_model,
        "Specialized": [egy_model_path, egy_vocab_path]
        if lazy_load_model
        else load_model(DiT, v1_base_cfg, egy_model_path, vocab_file=egy_vocab_path),
    },
    "MAR (Moroccan or Darija)": {
        "Unified": unified_model,
        "Specialized": [mar_model_path, mar_vocab_path]
        if lazy_load_model
        else load_model(DiT, v1_base_cfg, mar_model_path, vocab_file=mar_vocab_path),
    },
    "OMN (Omani, Dhofari, etc.)": {
        "Unified": unified_model,
    },
    "TUN (Tunisian)": {
        "Unified": unified_model,
    },
    "LEV (Levantine)": {
        "Unified": unified_model,
    },
    "SDN (Sudanese)": {
        "Unified": unified_model,
    },
    "LBY (Libyan)": {
        "Unified": unified_model,
    },
    "UNK (Unknown)": {
        "Unified": unified_model,
    },
}

# text-to-speech language reference example gallery
tts_lang_ref_examples_collections = {
    "MSA (Modern Standard Arabic)": [
        [
            files("habibi_tts").joinpath("assets/MSA.mp3"),
            "كان اللعيب حاضرًا في العديد من الأنشطة والفعاليات المرتبطة بكأس العالم، مما سمح للجماهير بالتفاعل معه والتقاط الصور التذكارية.",
        ],
    ],
    "SAU (Najdi, Hijazi, Gulf, etc.)": [
        [
            files("habibi_tts").joinpath("assets/Najdi.wav"),
            "تكفى طمني انا اليوم ماني بنايم ولا هو بداخل عيني النوم الين اتطمن عليه.",
        ],
        [
            files("habibi_tts").joinpath("assets/Hijazi.wav"),
            "ابغاك تحقق معاه بس بشكل ودي لانه سلطان يمر بظروف صعبة شوية.",
        ],
        [
            files("habibi_tts").joinpath("assets/Gulf.wav"),
            "وين تو الناس متى تصحى ومتى تفطر وتغير يبيلك ساعة يعني بالله تروح الشغل الساعة عشره.",
        ],
    ],
    "UAE (Emirati)": [
        [
            files("habibi_tts").joinpath("assets/UAE.wav"),
            "قمنا نشتريها بشكل متكرر أو لما نلقى ستايل يعجبنا وحياناً هذا الستايل ما نحبه.",
        ],
    ],
    "ALG (Algerian & Algerian Saharan)": [
        [files("habibi_tts").joinpath("assets/ALG.wav"), "أنيا هكا باغية ناكل هكا أني ن نشوف فيها الحاجة هذيكا."],
    ],
    "IRQ (Mesopotamian & North Mesopotamian)": [
        [
            files("habibi_tts").joinpath("assets/IRQ.wav"),
            "يعني ااا ما نقدر ناخذ وقت أكثر، ااا لأنه شروط كلش يحتاجلها وقت.",
        ],
    ],
    "EGY (Egyptian, Saidi, etc.)": [
        [
            files("habibi_tts").joinpath("assets/EGY.mp3"),
            "ايه الكلام. بقولك ايه. استخدم صوتي في المحادثات. استخدمه هيعجبك اوي.",
        ],
    ],
    "MAR (Moroccan or Darija)": [
        [
            files("habibi_tts").joinpath("assets/MAR.mp3"),
            "إذا بغيتي شي صوت باللهجة المغربية للإعلانات ديالك هذا أحسن واحد غادي تلقاه.",
        ],
    ],
    "OMN (Omani, Dhofari, etc.)": [],
    "TUN (Tunisian)": [],
    "LEV (Levantine)": [],
    "SDN (Sudanese)": [],
    "LBY (Libyan)": [],
    "UNK (Unknown)": [],
}

# default values (use the first in gallery as default)
default_tts_lang_choice = next(iter(tts_lang_model_collections))
default_tts_model_choice = next(iter(tts_lang_model_collections[default_tts_lang_choice]))
default_tts_ref_examples = tts_lang_ref_examples_collections[default_tts_lang_choice]

# define dropdown lists first, render later
choose_tts_lang = gr.Dropdown(
    choices=[t for t in tts_lang_model_collections],
    label="Choose TTS Language",
    value=default_tts_lang_choice,
)
choose_tts_model = gr.Dropdown(
    choices=[t for t in tts_lang_model_collections[default_tts_lang_choice]],
    label="Choose TTS Model",
    value=default_tts_model_choice,
)


@lru_cache(maxsize=1000)  # NOTE. need to ensure params of infer() hashable
@gpu_decorator
def infer(
    tts_lang_choice,
    tts_model_choice,
    ref_audio_orig,
    ref_text,
    gen_text,
    remove_silence=False,
    seed=-1,
    cross_fade_duration=0.15,
    nfe_step=32,
    speed=1,
    show_info=gr.Info,
):
    if not ref_audio_orig or not ref_text.strip() or not gen_text.strip():
        gr.Warning("Please ensure [Reference Audio] [Reference Text] [Text to Generate] are all provided.")
        return gr.update(), ref_text, seed

    if seed < 0 or seed > 2**31 - 1:
        gr.Warning("Please set a seed in range 0 ~ 2**31 - 1.")
        seed = np.random.randint(0, 2**31 - 1)
    torch.manual_seed(seed)
    used_seed = seed

    print(f"infer with [{tts_lang_choice}] [{tts_model_choice}] ...")

    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)

    if not lazy_load_model:
        model = tts_lang_model_collections[tts_lang_choice][tts_model_choice]
    else:
        model_path, vocab_path = tts_lang_model_collections[tts_lang_choice][tts_model_choice]
        model = load_model(
            DiT,
            v1_base_cfg,
            model_path,
            vocab_file=vocab_path,
        )

    if tts_model_choice == "Specialized":
        dialect_id = None
    elif tts_model_choice == "Unified":
        dialect_id = dialect_id_map[tts_lang_choice[:3]]
    else:
        raise AttributeError(f"[Code infer_gradio.py] unexpected tts_model_choice {tts_model_choice}")

    final_wave, final_sample_rate, _ = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
        show_info=show_info,
        progress=gr.Progress(),
        dialect_id=dialect_id,
    )

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(suffix=".wav", **tempfile_kwargs) as f:
            temp_path = f.name
        try:
            sf.write(temp_path, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        finally:
            os.unlink(temp_path)
        final_wave = final_wave.squeeze().cpu().numpy()

    return (final_sample_rate, final_wave), ref_text, used_seed


with gr.Blocks() as app_basic_tts:
    with gr.Row():
        with gr.Column():
            ref_wav_input = gr.Audio(label="Reference Audio", type="filepath")
            ref_txt_input = gr.Textbox(label="Reference Text")
            gen_txt_input = gr.Textbox(label="Text to Generate")
            generate_btn = gr.Button("Synthesize", variant="primary")
            with gr.Row():
                randomize_seed = gr.Checkbox(
                    label="Randomize Seed",
                    info="Check to use a random seed for each generation. Uncheck to use the seed specified.",
                    value=True,
                    scale=3,
                )
                seed_input = gr.Number(show_label=False, value=0, precision=0, scale=1)
        with gr.Column():
            audio_output = gr.Audio(label="Synthesized Audio")
            ref_examples = gr.Examples(
                examples=default_tts_ref_examples,
                inputs=[ref_wav_input, ref_txt_input],
                label="Example Prompts",
                examples_per_page=10,
            )

    def basic_tts(
        choose_tts_lang,
        choose_tts_model,
        ref_wav_input,
        ref_txt_input,
        gen_txt_input,
        randomize_seed,
        seed_input,
    ):
        if randomize_seed:
            seed_input = np.random.randint(0, 2**31 - 1)

        audio_out, ref_text_out, used_seed = infer(
            choose_tts_lang,
            choose_tts_model,
            ref_wav_input,
            ref_txt_input,
            gen_txt_input,
            seed=seed_input,
        )
        return audio_out, ref_text_out, used_seed

    generate_btn.click(
        basic_tts,
        inputs=[
            choose_tts_lang,
            choose_tts_model,
            ref_wav_input,
            ref_txt_input,
            gen_txt_input,
            randomize_seed,
            seed_input,
        ],
        outputs=[audio_output, ref_txt_input, seed_input],
    )


def load_text_from_file(file):
    if file:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    else:
        text = ""
    return gr.update(value=text)


def parse_speechtypes_text(gen_text):
    # Pattern to find {str} or {"name": str, "seed": int, "speed": float}
    pattern = r"(\{.*?\})"

    # Split the text by the pattern
    tokens = re.split(pattern, gen_text)

    segments = []

    current_type_dict = {
        "name": "Regular",
        "seed": -1,
        "speed": 1.0,
    }

    for i in range(len(tokens)):
        if i % 2 == 0:
            # This is text
            text = tokens[i].strip()
            if text:
                current_type_dict["text"] = text
                segments.append(current_type_dict)
        else:
            # This is type
            type_str = tokens[i].strip()
            try:  # if type dict
                current_type_dict = json.loads(type_str)
            except json.decoder.JSONDecodeError:
                type_str = type_str[1:-1]  # remove brace {}
                current_type_dict = {"name": type_str, "seed": -1, "speed": 1.0}

    return segments


with gr.Blocks() as app_multistyle:
    # New section for multistyle generation
    gr.Markdown(
        """
        # Multiple Speech-Type Generation

        This section allows you to generate multiple speech types or multiple people's voices. Enter your text in the format shown below, or upload a .txt file with the same format. The system will generate speech using the appropriate type. If unspecified, the model will use the regular speech type. The current speech type will be used until the next speech type is specified.
        """
    )

    with gr.Row():
        gr.Markdown(
            """
            **Example Input:** <br>
            {Speaker1_Emotion1} مرحبا يا صديقي. <br>
            {Speaker1_Emotion2} مرحبا يا صديقي. <br>
            {Speaker2_Emotion1} مرحبا يا صديقي. <br>
            {Speaker2_Emotion2} مرحبا يا صديقي. <br>
            """
        )

        gr.Markdown(
            """
            **Example Input 2:** <br>
            {"name": "Speaker1_Emotion1", "seed": -1, "speed": 1} مرحبا يا صديقي. <br>
            {"name": "Speaker1_Emotion2", "seed": -1, "speed": 1} مرحبا يا صديقي. <br>
            {"name": "Speaker2_Emotion1", "seed": -1, "speed": 1} مرحبا يا صديقي. <br>
            {"name": "Speaker2_Emotion2", "seed": -1, "speed": 1} مرحبا يا صديقي. <br>
            """
        )

    gr.Markdown(
        'Upload different audio clips for each speech type. The first speech type is mandatory. You can add additional speech types by clicking the "Add Speech Type" button.'
    )

    # Regular speech type (mandatory)
    with gr.Row(variant="compact") as regular_row:
        with gr.Column(scale=1, min_width=160):
            regular_name = gr.Textbox(value="Regular", label="Speech Type Name")
            regular_insert = gr.Button("Insert Label", variant="secondary")
        with gr.Column(scale=3):
            regular_audio = gr.Audio(label="Regular Reference Audio", type="filepath")
        with gr.Column(scale=3):
            regular_ref_text = gr.Textbox(label="Reference Text (Regular)", lines=4)
            with gr.Row():
                regular_seed_slider = gr.Slider(
                    show_label=False, minimum=-1, maximum=999, value=-1, step=1, info="Seed, -1 for random"
                )
                regular_speed_slider = gr.Slider(
                    show_label=False, minimum=0.3, maximum=2.0, value=1.0, step=0.1, info="Adjust the speed"
                )
        with gr.Column(scale=1, min_width=160):
            regular_ref_text_file = gr.File(label="Load Reference Text from File (.txt)", file_types=[".txt"])

    # Regular speech type (max 100)
    max_speech_types = 100
    speech_type_rows = [regular_row]
    speech_type_names = [regular_name]
    speech_type_audios = [regular_audio]
    speech_type_ref_texts = [regular_ref_text]
    speech_type_ref_text_files = [regular_ref_text_file]
    speech_type_seeds = [regular_seed_slider]
    speech_type_speeds = [regular_speed_slider]
    speech_type_delete_btns = [None]
    speech_type_insert_btns = [regular_insert]

    # Additional speech types (99 more)
    for i in range(max_speech_types - 1):
        with gr.Row(variant="compact", visible=False) as row:
            with gr.Column(scale=1, min_width=160):
                name_input = gr.Textbox(label="Speech Type Name")
                insert_btn = gr.Button("Insert Label", variant="secondary")
                delete_btn = gr.Button("Delete Type", variant="stop")
            with gr.Column(scale=3):
                audio_input = gr.Audio(label="Reference Audio", type="filepath")
            with gr.Column(scale=3):
                ref_text_input = gr.Textbox(label="Reference Text", lines=4)
                with gr.Row():
                    seed_input = gr.Slider(
                        show_label=False, minimum=-1, maximum=999, value=-1, step=1, info="Seed. -1 for random"
                    )
                    speed_input = gr.Slider(
                        show_label=False, minimum=0.3, maximum=2.0, value=1.0, step=0.1, info="Adjust the speed"
                    )
            with gr.Column(scale=1, min_width=160):
                ref_text_file_input = gr.File(label="Load Reference Text from File (.txt)", file_types=[".txt"])
        speech_type_rows.append(row)
        speech_type_names.append(name_input)
        speech_type_audios.append(audio_input)
        speech_type_ref_texts.append(ref_text_input)
        speech_type_ref_text_files.append(ref_text_file_input)
        speech_type_seeds.append(seed_input)
        speech_type_speeds.append(speed_input)
        speech_type_delete_btns.append(delete_btn)
        speech_type_insert_btns.append(insert_btn)

    # Global logic for all speech types
    for i in range(max_speech_types):
        speech_type_audios[i].clear(
            lambda: [None, None],
            None,
            [speech_type_ref_texts[i], speech_type_ref_text_files[i]],
        )
        speech_type_ref_text_files[i].upload(
            load_text_from_file,
            inputs=[speech_type_ref_text_files[i]],
            outputs=[speech_type_ref_texts[i]],
        )

    # Button to add speech type
    add_speech_type_btn = gr.Button("Add Speech Type")

    # Keep track of autoincrement of speech types, no roll back
    speech_type_count = 1

    # Function to add a speech type
    def add_speech_type_fn():
        row_updates = [gr.update() for _ in range(max_speech_types)]
        global speech_type_count
        if speech_type_count < max_speech_types:
            row_updates[speech_type_count] = gr.update(visible=True)
            speech_type_count += 1
        else:
            gr.Warning("Exhausted maximum number of speech types. Consider restart the app.")
        return row_updates

    add_speech_type_btn.click(add_speech_type_fn, outputs=speech_type_rows)

    # Function to delete a speech type
    def delete_speech_type_fn():
        return gr.update(visible=False), None, None, None, None

    # Update delete button clicks and ref text file changes
    for i in range(1, len(speech_type_delete_btns)):
        speech_type_delete_btns[i].click(
            delete_speech_type_fn,
            outputs=[
                speech_type_rows[i],
                speech_type_names[i],
                speech_type_audios[i],
                speech_type_ref_texts[i],
                speech_type_ref_text_files[i],
            ],
        )

    # Text input for the prompt
    with gr.Row():
        gen_text_input_multistyle = gr.Textbox(
            label="Text to Generate",
            lines=10,
            max_lines=40,
            scale=4,
            placeholder="Enter the script with speaker names (or emotion types) at the start of each block, e.g.:\n\n{Speaker1_Emotion1} مرحبا يا صديقي.\n{Speaker1_Emotion2} مرحبا يا صديقي.\n{Speaker2_Emotion1} مرحبا يا صديقي.\n{Speaker2_Emotion2} مرحبا يا صديقي.",
        )
        gen_text_file_multistyle = gr.File(label="Load Text to Generate from File (.txt)", file_types=[".txt"], scale=1)

    def make_insert_speech_type_fn(index):
        def insert_speech_type_fn(current_text, speech_type_name, speech_type_seed, speech_type_speed):
            current_text = current_text or ""
            if not speech_type_name:
                gr.Warning("Please enter speech type name before insert.")
                return current_text
            speech_type_dict = {
                "name": speech_type_name,
                "seed": speech_type_seed,
                "speed": speech_type_speed,
            }
            updated_text = current_text + json.dumps(speech_type_dict) + " "
            return updated_text

        return insert_speech_type_fn

    for i, insert_btn in enumerate(speech_type_insert_btns):
        insert_fn = make_insert_speech_type_fn(i)
        insert_btn.click(
            insert_fn,
            inputs=[gen_text_input_multistyle, speech_type_names[i], speech_type_seeds[i], speech_type_speeds[i]],
            outputs=gen_text_input_multistyle,
        )

    with gr.Accordion("Advanced Settings", open=True):
        with gr.Row():
            with gr.Column():
                show_cherrypick_multistyle = gr.Checkbox(
                    label="Show Cherry-pick Interface",
                    info="Turn on to show interface, picking seeds from previous generations.",
                    value=False,
                )
            with gr.Column():
                remove_silence_multistyle = gr.Checkbox(
                    label="Remove Silences",
                    info="Turn on to automatically detect and crop long silences.",
                    value=True,
                )

    # Generate button
    generate_multistyle_btn = gr.Button("Generate Multi-Style Speech", variant="primary")

    # Output audio
    audio_output_multistyle = gr.Audio(label="Synthesized Audio")

    # Used seed gallery
    cherrypick_interface_multistyle = gr.Textbox(
        label="Cherry-pick Interface",
        lines=10,
        max_lines=40,
        buttons=["copy"],  # show_copy_button=True if gradio<6.0
        interactive=False,
        visible=False,
    )

    # Logic control to show/hide the cherrypick interface
    show_cherrypick_multistyle.change(
        lambda is_visible: gr.update(visible=is_visible),
        show_cherrypick_multistyle,
        cherrypick_interface_multistyle,
    )

    # Function to load text to generate from file
    gen_text_file_multistyle.upload(
        load_text_from_file,
        inputs=[gen_text_file_multistyle],
        outputs=[gen_text_input_multistyle],
    )

    @gpu_decorator
    def generate_multistyle_speech(
        choose_tts_lang,
        choose_tts_model,
        gen_text,
        *args,
    ):
        speech_type_names_list = args[:max_speech_types]
        speech_type_audios_list = args[max_speech_types : 2 * max_speech_types]
        speech_type_ref_texts_list = args[2 * max_speech_types : 3 * max_speech_types]
        remove_silence = args[3 * max_speech_types]
        # Collect the speech types and their audios into a dict
        speech_types = OrderedDict()

        ref_text_idx = 0
        for name_input, audio_input, ref_text_input in zip(
            speech_type_names_list, speech_type_audios_list, speech_type_ref_texts_list
        ):
            if name_input and audio_input:
                speech_types[name_input] = {"audio": audio_input, "ref_text": ref_text_input}
            else:
                speech_types[f"@{ref_text_idx}@"] = {"audio": "", "ref_text": ""}
            ref_text_idx += 1

        # Parse the gen_text into segments
        segments = parse_speechtypes_text(gen_text)

        # For each segment, generate speech
        generated_audio_segments = []
        current_type_name = "Regular"
        inference_meta_data = ""

        for segment in segments:
            name = segment["name"]
            seed_input = segment["seed"]
            speed = segment["speed"]
            text = segment["text"]

            if name in speech_types:
                current_type_name = name
            else:
                gr.Warning(f"Type {name} is not available, will use Regular as default.")
                current_type_name = "Regular"

            try:
                ref_audio = speech_types[current_type_name]["audio"]
            except KeyError:
                gr.Warning(f"Please provide reference audio for type {current_type_name}.")
                return [None] + [speech_types[name]["ref_text"] for name in speech_types] + [None]
            ref_text = speech_types[current_type_name].get("ref_text", "")

            if seed_input == -1:
                seed_input = np.random.randint(0, 2**31 - 1)

            # Generate or retrieve speech for this segment
            audio_out, ref_text_out, used_seed = infer(
                choose_tts_lang,
                choose_tts_model,
                ref_audio,
                ref_text,
                text,
                remove_silence,
                seed=seed_input,
                cross_fade_duration=0,
                speed=speed,
                show_info=print,  # no pull to top when generating
            )
            sr, audio_data = audio_out

            generated_audio_segments.append(audio_data)
            speech_types[current_type_name]["ref_text"] = ref_text_out
            inference_meta_data += json.dumps(dict(name=name, seed=used_seed, speed=speed)) + f" {text}\n"

        # Concatenate all audio segments
        if generated_audio_segments:
            final_audio_data = np.concatenate(generated_audio_segments)
            return (
                [(sr, final_audio_data)]
                + [speech_types[name]["ref_text"] for name in speech_types]
                + [inference_meta_data]
            )
        else:
            gr.Warning("No audio generated.")
            return [None] + [speech_types[name]["ref_text"] for name in speech_types] + [None]

    generate_multistyle_btn.click(
        generate_multistyle_speech,
        inputs=[
            choose_tts_lang,
            choose_tts_model,
            gen_text_input_multistyle,
        ]
        + speech_type_names
        + speech_type_audios
        + speech_type_ref_texts
        + [
            remove_silence_multistyle,
        ],
        outputs=[audio_output_multistyle] + speech_type_ref_texts + [cherrypick_interface_multistyle],
    )

    # Validation function to disable Generate button if speech types are missing
    def validate_speech_types(gen_text, regular_name, *args):
        speech_type_names_list = args

        # Collect the speech types names
        speech_types_available = set()
        if regular_name:
            speech_types_available.add(regular_name)
        for name_input in speech_type_names_list:
            if name_input:
                speech_types_available.add(name_input)

        # Parse the gen_text to get the speech types used
        segments = parse_speechtypes_text(gen_text)
        speech_types_in_text = set(segment["name"] for segment in segments)

        # Check if all speech types in text are available
        missing_speech_types = speech_types_in_text - speech_types_available

        if missing_speech_types:
            # Disable the generate button
            return gr.update(interactive=False)
        else:
            # Enable the generate button
            return gr.update(interactive=True)

    gen_text_input_multistyle.change(
        validate_speech_types,
        inputs=[gen_text_input_multistyle, regular_name] + speech_type_names,
        outputs=generate_multistyle_btn,
    )


with gr.Blocks() as app:
    gr.Markdown(
        f"""
        # [Habibi](https://arxiv.org/abs/2601.13802): Laying the Open-Source Foundation of Unified-Dialectal Arabic Speech Synthesis

        This is {"a local web UI" if not USING_SPACES else "an online demo"} for [Habibi-TTS](https://github.com/SWivid/Habibi-TTS).

        Several notes:

        * Use more qualified reference audio instead of default, ensure less than 12 seconds (use  ✂  to clip)
        * Provide an exactly matched reference text, which ends with proper punctuation, e.g. a period
        * Ensure the audio is fully uploaded before generating
        * If any issues, try convert to WAV or MP3 format first{", and check FFmpeg installation" if not USING_SPACES else ""}

        **The terminology (three letters capitalized) does not reflect any official classification for Arabic.**
        """
    )

    def switch_tts_lang(new_lang_choice):
        new_tts_model_choice = next(iter(tts_lang_model_collections[new_lang_choice]))  # first as default
        new_tts_ref_examples = tts_lang_ref_examples_collections[new_lang_choice]
        return gr.update(
            choices=[t for t in tts_lang_model_collections[new_lang_choice]], value=new_tts_model_choice
        ), gr.Dataset(samples=new_tts_ref_examples)

    with gr.Row():
        choose_tts_lang.render()
        choose_tts_model.render()

    choose_tts_lang.change(
        switch_tts_lang,
        inputs=[choose_tts_lang],
        outputs=[choose_tts_model, ref_examples.dataset],
    )

    gr.TabbedInterface(
        [app_basic_tts, app_multistyle],
        ["Basic-TTS", "Multi-Speech"],
    )


@click.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
@click.option("--host", "-H", default=None, help="Host to run the app on")
@click.option(
    "--share",
    "-s",
    default=False,
    is_flag=True,
    help="Share the app via Gradio share link",
)
@click.option("--api", "-a", default=True, is_flag=True, help="Allow API access")
@click.option(
    "--root_path",
    "-r",
    default=None,
    type=str,
    help='The root path (or "mount point") of the application, if it\'s not served from the root ("/") of the domain. Often used when the application is behind a reverse proxy that forwards requests to the application, e.g. set "/myapp" or full URL for application served at "https://example.com/myapp".',
)
@click.option(
    "--inbrowser",
    "-i",
    is_flag=True,
    default=False,
    help="Automatically launch the interface in the default web browser",
)
def main(port, host, share, api, root_path, inbrowser):
    global app
    print("Starting app...")
    app.queue(api_open=api).launch(
        server_name=host,
        server_port=port,
        share=share,
        root_path=root_path,
        inbrowser=inbrowser,
    )


if __name__ == "__main__":
    if not USING_SPACES:
        main()
    else:
        app.queue().launch()
