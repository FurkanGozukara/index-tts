import html
import json
import os
import sys
import threading
import time
import glob

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(
    description="IndexTTS WebUI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
parser.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed to accelerate if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available")
parser.add_argument("--gui_seg_tokens", type=int, default=120, help="GUI: Max tokens per generation segment")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt"
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import gradio as gr
from indextts.infer_v2 import IndexTTS2
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="Auto")
MODE = 'local'
tts = IndexTTS2(model_dir=cmd_args.model_dir,
                cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
                use_fp16=cmd_args.fp16,
                use_deepspeed=cmd_args.deepspeed,
                use_cuda_kernel=cmd_args.cuda_kernel,
                )
# 支持的语言列表
LANGUAGES = {
    "中文": "zh_CN",
    "English": "en_US"
}
EMO_CHOICES_ALL = ["Same as speaker voice",
                "Use emotion reference audio",
                "Use emotion vector control",
                "Use emotion text description"]

os.makedirs("outputs/tasks",exist_ok=True)
os.makedirs("prompts",exist_ok=True)

MAX_LENGTH_TO_USE_SPEED = 70

# Try to import pydub for MP3 export
try:
    from pydub import AudioSegment
    MP3_AVAILABLE = True
except ImportError:
    MP3_AVAILABLE = False
    print("Warning: pydub not installed. MP3 export will not be available.")
    print("To enable MP3 export, install pydub: pip install pydub")

def get_next_file_number(output_dir="outputs", target_folder=None, prefix=""):
    """Get the next available file number in sequence."""
    if target_folder:
        output_dir = target_folder

    os.makedirs(output_dir, exist_ok=True)

    # Find all existing files with our naming pattern
    existing_files = glob.glob(os.path.join(output_dir, f"{prefix}[0-9][0-9][0-9][0-9].*"))

    if not existing_files:
        return 1

    # Extract numbers from filenames
    numbers = []
    for filepath in existing_files:
        filename = os.path.basename(filepath)
        # Remove prefix if present
        if prefix:
            filename = filename[len(prefix):]
        # Extract the 4-digit number
        try:
            num_str = filename[:4]
            if num_str.isdigit():
                numbers.append(int(num_str))
        except:
            continue

    if numbers:
        return max(numbers) + 1
    else:
        return 1

def generate_output_path(target_folder=None, filename=None, save_as_mp3=False, prefix=""):
    """Generate output file path with sequential numbering."""
    output_dir = target_folder if target_folder else "outputs"
    os.makedirs(output_dir, exist_ok=True)

    if filename:
        # Use provided filename
        extension = ".mp3" if save_as_mp3 and MP3_AVAILABLE else ".wav"
        if not filename.endswith(('.wav', '.mp3')):
            filename = filename + extension
        return os.path.join(output_dir, filename)
    else:
        # Use sequential numbering
        next_num = get_next_file_number(output_dir, target_folder, prefix)
        extension = ".mp3" if save_as_mp3 and MP3_AVAILABLE else ".wav"
        filename = f"{prefix}{next_num:04d}{extension}"
        return os.path.join(output_dir, filename)

def convert_wav_to_mp3(wav_path, mp3_path, bitrate="256k"):
    """Convert WAV file to MP3 using pydub."""
    if not MP3_AVAILABLE:
        print("Warning: MP3 conversion not available. Keeping WAV format.")
        return wav_path

    try:
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format="mp3", bitrate=bitrate)
        # Remove the original WAV file
        os.remove(wav_path)
        return mp3_path
    except Exception as e:
        print(f"Error converting to MP3: {e}")
        return wav_path

def gen_single(emo_control_method,prompt, text,
               emo_ref_path, emo_weight,
               vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
               emo_text,emo_random,
               max_text_tokens_per_segment,
               save_as_mp3,
               # Expert params (in order from expert_params list)
               diffusion_steps,
               inference_cfg_rate,
               interval_silence,
               max_speaker_audio_length,
               max_emotion_audio_length,
               autoregressive_batch_size,
               apply_emo_bias,
               max_emotion_sum,
               latent_multiplier,
               max_consecutive_silence,
               mp3_bitrate,
               # Advanced params (in order from advanced_params list)
               do_sample,
               top_p,
               top_k,
               temperature,
               length_penalty,
               num_beams,
               repetition_penalty,
               max_mel_tokens,
               low_memory_mode,
               progress=gr.Progress()):
    # Generate output path with sequential numbering
    temp_wav_path = generate_output_path(save_as_mp3=False)  # Always generate WAV first
    output_path = temp_wav_path
    # set gradio progress
    tts.gr_progress = progress

    # Update the low memory mode setting
    tts.hybrid_model_device = bool(low_memory_mode)

    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
        # "typical_sampling": bool(typical_sampling),
        # "typical_mass": float(typical_mass),
    }
    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value
    if emo_control_method == 0:  # emotion from speaker
        emo_ref_path = None  # remove external reference audio
    if emo_control_method == 1:  # emotion from reference audio
        pass
    if emo_control_method == 2:  # emotion from custom vectors
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        vec = tts.normalize_emo_vec(vec, apply_bias=apply_emo_bias, max_emotion_sum=max_emotion_sum)
    else:
        # don't use the emotion vector inputs for the other modes
        vec = None

    if emo_text == "":
        # erase empty emotion descriptions; `infer()` will then automatically use the main prompt
        emo_text = None

    print(f"Emo control mode:{emo_control_method},weight:{emo_weight},vec:{vec}")

    # Pass new parameters to infer
    output = tts.infer(spk_audio_prompt=prompt, text=text,
                       output_path=output_path,
                       emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                       emo_vector=vec,
                       use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                       verbose=cmd_args.verbose,
                       max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                       interval_silence=int(interval_silence),
                       diffusion_steps=int(diffusion_steps),
                       inference_cfg_rate=float(inference_cfg_rate),
                       max_speaker_audio_length=float(max_speaker_audio_length),
                       max_emotion_audio_length=float(max_emotion_audio_length),
                       autoregressive_batch_size=int(autoregressive_batch_size),
                       max_emotion_sum=float(max_emotion_sum),
                       latent_multiplier=float(latent_multiplier),
                       max_consecutive_silence=int(max_consecutive_silence),
                       **kwargs)

    # Convert to MP3 if requested
    if save_as_mp3 and MP3_AVAILABLE:
        mp3_path = output.replace('.wav', '.mp3')
        output = convert_wav_to_mp3(output, mp3_path, bitrate=mp3_bitrate)

    return gr.update(value=output,visible=True)

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button


theme = gr.themes.Soft()
theme.font = [gr.themes.GoogleFont("Inter"), "Tahoma", "ui-sans-serif", "system-ui", "sans-serif"]
with gr.Blocks(title="SECourses IndexTTS2 Premium App", theme=theme) as demo:
    mutex = threading.Lock()
    gr.Markdown("## SECourses Index TTS2 Premium App : https://www.patreon.com/c/SECourses")

    with gr.Tab("Audio Generation"):
        with gr.Row():
            os.makedirs("prompts",exist_ok=True)
            prompt_audio = gr.Audio(
                label="Speaker Reference Audio (3-15 seconds)",
                key="prompt_audio",
                sources=["upload","microphone"],
                type="filepath"
            )
            prompt_list = os.listdir("prompts")
            default = ''
            if prompt_list:
                default = prompt_list[0]
            with gr.Column():
                input_text_single = gr.TextArea(
                    label="Text to Synthesize",
                    key="input_text_single",
                    placeholder="Enter the text you want to convert to speech",
                    info=f"Model v{tts.model_version or '1.0'} | Supports multiple languages. Long texts will be automatically segmented."
                )
                gen_button = gr.Button("Generate Speech", key="gen_button", interactive=True, variant="primary")
            output_audio = gr.Audio(
                label="Generated Result (click to play/download)",
                visible=True,
                key="output_audio"
            )

        with gr.Accordion("Function Settings"):
            # 情感控制选项部分 - now showing ALL options including experimental
            with gr.Row():
                emo_control_method = gr.Radio(
                    choices=EMO_CHOICES_ALL,
                    type="index",
                    value=EMO_CHOICES_ALL[0],
                    label="Emotion Control Method",
                    info="Choose how to control emotions: Speaker's natural emotion, reference audio emotion, manual vector control, or text description"
                )
        # 情感参考音频部分
        with gr.Group(visible=False) as emotion_reference_group:
            with gr.Row():
                emo_upload = gr.Audio(
                    label="Upload Emotion Reference Audio",
                    type="filepath"
                )

        # 情感随机采样
        with gr.Row(visible=False) as emotion_randomize_group:
            emo_random = gr.Checkbox(
                label="Random Emotion Sampling",
                value=False,
                info="Enable random sampling from emotion matrix for more varied emotional expression"
            )

        # 情感向量控制部分
        with gr.Group(visible=False) as emotion_vector_group:
            with gr.Row():
                with gr.Column():
                    vec1 = gr.Slider(label="Joy", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Happiness and cheerfulness in voice")
                    vec2 = gr.Slider(label="Anger", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Aggressive and forceful tone")
                    vec3 = gr.Slider(label="Sadness", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Melancholic and sorrowful expression")
                    vec4 = gr.Slider(label="Fear", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Anxious and worried tone")
                with gr.Column():
                    vec5 = gr.Slider(label="Disgust", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Repulsed and disgusted expression")
                    vec6 = gr.Slider(label="Depression", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Low energy and melancholic mood")
                    vec7 = gr.Slider(label="Surprise", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Shocked and amazed reaction")
                    vec8 = gr.Slider(label="Calm", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Neutral and peaceful tone")

        with gr.Group(visible=False) as emo_text_group:
            with gr.Row():
                emo_text = gr.Textbox(label="Emotion Description Text",
                                      placeholder="Enter emotion description (or leave empty to automatically use target text as emotion description)",
                                      value="",
                                      info="e.g.: feeling wronged, danger is approaching quietly")

        with gr.Row(visible=False) as emo_weight_group:
            emo_weight = gr.Slider(
                label="Emotion Weight",
                minimum=0.0,
                maximum=1.0,
                value=0.65,
                step=0.01,
                info="Controls the strength of emotion blending. 0 = no emotion, 1 = full emotion from reference. Default: 0.65"
            )

        with gr.Accordion("Advanced Generation Parameter Settings", open=False, visible=True) as advanced_settings_group:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"**GPT2 Sampling Settings** _Parameters affect audio diversity and generation speed. See [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)._")
                    with gr.Row():
                        do_sample = gr.Checkbox(
                            label="Enable Sampling",
                            value=True,
                            info="Use probabilistic sampling for more natural variation. Turn OFF for deterministic output."
                        )
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.1,
                            maximum=2.0,
                            value=0.8,
                            step=0.1,
                            info="Controls randomness. Lower = more focused/predictable, Higher = more creative/diverse. Default: 0.8"
                        )
                    with gr.Row():
                        top_p = gr.Slider(
                            label="Top-p (Nucleus Sampling)",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.8,
                            step=0.01,
                            info="Cumulative probability for token selection. Lower = safer choices, Higher = more variation. Default: 0.8"
                        )
                        top_k = gr.Slider(
                            label="Top-k",
                            minimum=0,
                            maximum=100,
                            value=30,
                            step=1,
                            info="Number of highest probability tokens to consider. 0 = disabled. Default: 30"
                        )
                        num_beams = gr.Slider(
                            label="Beam Search Beams",
                            value=3,
                            minimum=1,
                            maximum=10,
                            step=1,
                            info="Number of beams for beam search. Higher = better quality but slower. Default: 3"
                        )
                    with gr.Row():
                        repetition_penalty = gr.Number(
                            label="Repetition Penalty",
                            precision=None,
                            value=10.0,
                            minimum=0.1,
                            maximum=20.0,
                            step=0.1,
                            info="Penalizes repeated tokens. Higher = less repetition. Default: 10.0"
                        )
                        length_penalty = gr.Number(
                            label="Length Penalty",
                            precision=None,
                            value=0.0,
                            minimum=-2.0,
                            maximum=2.0,
                            step=0.1,
                            info="Controls output length. Positive = longer, Negative = shorter. Default: 0.0"
                        )
                        save_as_mp3 = gr.Checkbox(label="Save as MP3", value=False,
                                                  visible=MP3_AVAILABLE,
                                                  info="Save audio as MP3 format instead of WAV" if MP3_AVAILABLE else "Requires pydub: pip install pydub")
                    max_mel_tokens = gr.Slider(label="max_mel_tokens", value=1500, minimum=50, maximum=tts.cfg.gpt.max_mel_tokens, step=10, info="Maximum number of generated tokens. Too small will cause audio truncation", key="max_mel_tokens")
                    # with gr.Row():
                    #     typical_sampling = gr.Checkbox(label="typical_sampling", value=False, info="不建议使用")
                    #     typical_mass = gr.Slider(label="typical_mass", value=0.9, minimum=0.0, maximum=1.0, step=0.1)
                with gr.Column(scale=2):
                    gr.Markdown(f'**Sentence Segmentation Settings** _Parameters affect audio quality and generation speed_')
                    with gr.Row():
                        with gr.Column():
                            initial_value = max(20, min(tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens))
                            max_text_tokens_per_segment = gr.Slider(
                                label="Max Tokens per Segment", value=initial_value, minimum=20, maximum=tts.cfg.gpt.max_text_tokens, step=2, key="max_text_tokens_per_segment",
                                info="Recommended range: 80-200. Larger values = longer segments; smaller values = more fragmented. Too small or too large may reduce audio quality",
                            )
                        with gr.Column():
                            low_memory_mode = gr.Checkbox(label="Low Memory Mode", value=False,
                                                         info="Enable low memory mode for systems with limited GPU memory (inference will be slower)")
                    with gr.Accordion("Preview Sentence Segmentation Results", open=True) as segments_settings:
                        segments_preview = gr.Dataframe(
                            headers=["Index", "Segment Content", "Token Count"],
                            key="segments_preview",
                            wrap=True,
                        )

        with gr.Accordion("Expert Diffusion & Audio Processing Settings", open=False) as expert_settings_group:
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**🔬 Diffusion Model Parameters**")
                    diffusion_steps = gr.Slider(
                        label="Diffusion Steps",
                        value=25,
                        minimum=10,
                        maximum=100,
                        step=1,
                        info="Number of denoising steps in the diffusion model. Higher = better quality but slower. Default: 25"
                    )
                    inference_cfg_rate = gr.Slider(
                        label="CFG Rate (Classifier-Free Guidance)",
                        value=0.7,
                        minimum=0.0,
                        maximum=2.0,
                        step=0.05,
                        info="Controls how strongly the model follows the conditioning. 0 = no guidance, 1 = full guidance. Default: 0.7"
                    )
                    latent_multiplier = gr.Slider(
                        label="Latent Length Multiplier",
                        value=1.72,
                        minimum=1.0,
                        maximum=3.0,
                        step=0.01,
                        info="Multiplier for target audio length calculation. Affects speech pacing. Default: 1.72"
                    )
                with gr.Column():
                    gr.Markdown("**🎵 Audio Processing Parameters**")
                    interval_silence = gr.Slider(
                        label="Silence Between Segments (ms)",
                        value=200,
                        minimum=0,
                        maximum=1000,
                        step=50,
                        info="Milliseconds of silence inserted between text segments. Default: 200ms"
                    )
                    max_consecutive_silence = gr.Slider(
                        label="Max Consecutive Silent Tokens",
                        value=30,
                        minimum=10,
                        maximum=100,
                        step=5,
                        info="Maximum allowed consecutive silent tokens before compression. Reduces long pauses. Default: 30"
                    )
                    mp3_bitrate = gr.Dropdown(
                        label="MP3 Bitrate",
                        choices=["128k", "192k", "256k", "320k"],
                        value="256k",
                        info="Audio quality for MP3 export. Higher = better quality, larger file. Default: 256k"
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**📊 Reference Audio Limits**")
                    max_speaker_audio_length = gr.Slider(
                        label="Max Speaker Reference Length (seconds)",
                        value=15,
                        minimum=5,
                        maximum=60,
                        step=1,
                        info="Maximum duration for speaker reference audio. Longer clips will be truncated. Default: 15s"
                    )
                    max_emotion_audio_length = gr.Slider(
                        label="Max Emotion Reference Length (seconds)",
                        value=15,
                        minimum=5,
                        maximum=60,
                        step=1,
                        info="Maximum duration for emotion reference audio. Longer clips will be truncated. Default: 15s"
                    )
                with gr.Column():
                    gr.Markdown("**🎯 Emotion Control Fine-tuning**")
                    apply_emo_bias = gr.Checkbox(
                        label="Apply Emotion Bias Correction",
                        value=True,
                        info="Apply automatic bias to prevent extreme emotion outputs. Recommended: ON"
                    )
                    max_emotion_sum = gr.Slider(
                        label="Max Total Emotion Strength",
                        value=0.8,
                        minimum=0.1,
                        maximum=2.0,
                        step=0.05,
                        info="Maximum sum of all emotion vectors. Prevents over-emotional speech. Default: 0.8"
                    )
                    autoregressive_batch_size = gr.Slider(
                        label="Autoregressive Batch Size",
                        value=1,
                        minimum=1,
                        maximum=4,
                        step=1,
                        info="Batch size for autoregressive generation. Higher may improve diversity. Default: 1"
                    )

            advanced_params = [
                do_sample, top_p, top_k, temperature,
                length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                low_memory_mode,
                # typical_sampling, typical_mass,
            ]

            expert_params = [
                diffusion_steps, inference_cfg_rate, interval_silence,
                max_speaker_audio_length, max_emotion_audio_length,
                autoregressive_batch_size, apply_emo_bias, max_emotion_sum,
                latent_multiplier, max_consecutive_silence, mp3_bitrate
            ]



    def on_input_text_change(text, max_text_tokens_per_segment):
        if text and len(text) > 0:
            text_tokens_list = tts.tokenizer.tokenize(text)

            segments = tts.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment=int(max_text_tokens_per_segment))
            data = []
            for i, s in enumerate(segments):
                segment_str = ''.join(s)
                tokens_count = len(s)
                data.append([i, segment_str, tokens_count])
            return {
                segments_preview: gr.update(value=data, visible=True, type="array"),
            }
        else:
            df = pd.DataFrame([], columns=["Index", "Segment Content", "Token Count"])
            return {
                segments_preview: gr.update(value=df),
            }

    def on_method_change(emo_control_method):
        if emo_control_method == 1:  # emotion reference audio
            return (gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 2:  # emotion vectors
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 3:  # emotion text description
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True)
                    )
        else:  # 0: same as speaker voice
            return (gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                    )

    emo_control_method.change(on_method_change,
        inputs=[emo_control_method],
        outputs=[emotion_reference_group,
                 emotion_randomize_group,
                 emotion_vector_group,
                 emo_text_group,
                 emo_weight_group]
    )


    input_text_single.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )

    max_text_tokens_per_segment.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )

    prompt_audio.upload(update_prompt_audio,
                         inputs=[],
                         outputs=[gen_button])

    gen_button.click(gen_single,
                     inputs=[emo_control_method,prompt_audio, input_text_single, emo_upload, emo_weight,
                            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                             emo_text,emo_random,
                             max_text_tokens_per_segment,
                             save_as_mp3,
                             *expert_params,
                             *advanced_params,
                     ],
                     outputs=[output_audio])



if __name__ == "__main__":
    demo.queue(20)
    demo.launch(inbrowser=True)
