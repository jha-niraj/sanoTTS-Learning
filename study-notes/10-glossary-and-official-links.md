# 10 — Glossary + Every Official Link

## Part A — Glossary (alphabetical)

| Term | Plain meaning | Where you meet it |
|---|---|---|
| **Acoustic model / acoustic student** | predicts what each moment of speech sounds like; in sanoTTS, the 192-number latent per frame | file 05, `train_roota_piper_latent_student.py` |
| **Activation** | (1) the "bend" function in a network (SiLU, LeakyReLU); (2) the intermediate values flowing through a network | file 01 |
| **Adversarial / GAN** | training with a second network (discriminator) that tries to spot fakes | file 06 |
| **AdamW** | the optimizer used everywhere here; adaptive step sizes | file 01 |
| **Autograd** | PyTorch automatically computing gradients via `loss.backward()` | file 01 |
| **Batch** | a small group of examples processed together in one training step | file 01 |
| **BOS / EOS / PAD** | begin / end / padding symbols; Piper IDs 1 / 2 / 0 | file 02 |
| **Checkpoint (.pt)** | saved model weights + config | file 01 |
| **CER / WER** | character / word error rate: how many edits turn the ASR transcript into the reference | file 06 |
| **Channel** | one of the "rows" of numbers per time step (e.g. 192 latent channels) | file 05 |
| **Conv1d** | a sliding-window layer over time | file 01 |
| **ConvTranspose1d** | an upsampling layer that stretches time | file 05 |
| **CTC** | a loss/decoding method for ASR that aligns audio frames to characters without explicit alignment | file 09 |
| **Decoder student / vocoder** | turns features (latent) into waveform samples; "piperlite" in sanoTTS | file 05 |
| **Decoder cut** | the teacher's decoder extracted as its own ONNX: latent → audio | file 04 |
| **Dilation** | a convolution that skips samples, to see farther with the same number of weights | file 05 |
| **Discriminator** | the "critic" network in a GAN; training-only in sanoTTS | file 06 |
| **Distillation** | training a small student to imitate a big teacher | file 03 |
| **Duration model / student** | predicts how many frames each phoneme lasts | file 05 |
| **Embedding** | a lookup table from an ID to a learned vector | file 01 |
| **espeak-ng** | rule-based open-source speech engine; used here only for G2P | file 02 |
| **Eval / held-out set** | data never used for training; used to measure honestly | file 01 |
| **Feature matching** | making the discriminator's inner layers respond to fake audio like real audio | file 06 |
| **fp32 / fp16 / int8** | 4-byte float / 2-byte float / 1-byte integer storage for numbers | files 01, 07 |
| **Forward / backward pass** | run the model / compute gradients | file 01 |
| **Frame** | a time slice; here 256 samples = 11.6 ms | file 02 |
| **G2P** | grapheme-to-phoneme: letters to sound symbols | file 02 |
| **generator_input** | the teacher's 192-channel latent entering its decoder; *the* interface sanoTTS copies | files 02, 03 |
| **Golden vectors / golden gate** | saved reference outputs; ports must match them (correlation ≥ 0.98) | file 07 |
| **Gradient** | the direction each knob should move to reduce loss | file 01 |
| **HiFi-GAN** | a GAN vocoder architecture; Piper's decoder is one | file 02 |
| **Hop** | distance between frames (256 samples here) | file 02 |
| **Inference** | using a trained model (as opposed to training it) | file 01 |
| **IPA** | International Phonetic Alphabet | file 02 |
| **iSTFT** | inverse STFT: spectrum back to waveform (used by the `heart` / MCU decoders) | file 07 |
| **Joint finetune** | training acoustic + decoder together at the end | file 04 |
| **Lanes** | rendering the same text through teacher/student combinations to find which part is weak | file 06 |
| **Latent** | an internal compressed representation learned by a network | file 03 |
| **Learning rate (lr)** | how big each training nudge is | file 01 |
| **length_scale** | a multiplier on predicted durations; >1 = slower speech, no retraining needed | file 05 |
| **Loss** | a single number measuring how wrong the model is | file 01 |
| **Mask** | True/False grid marking real vs padding positions | file 01 |
| **Mel spectrogram** | a spectrogram warped to match human pitch perception | file 02 |
| **MFA** | Montreal Forced Aligner: finds when each word/phoneme occurs in a recording | file 08 |
| **MOS / SCOREQ / UTMOS / DNSMOS** | human quality score (1–5) / neural predictors of it | file 06 |
| **Multi-resolution STFT loss** | comparing spectrograms at several window sizes | file 06 |
| **ONNX / ONNX Runtime** | portable model graph format / the engine that runs it | files 01, 03 |
| **Optimizer** | the algorithm that applies gradients to update weights | file 01 |
| **Overfitting / memorization** | great on training data, bad on new data | file 01 |
| **Pack (teacher pack)** | saved teacher outputs (IDs, durations, latents, audio) for many sentences | file 03 |
| **Parameter / weight** | one learnable number in a model | file 01 |
| **Parity** | proof that two implementations compute the same thing | file 04 |
| **Phoneme** | a unit of sound in a language | file 02 |
| **Piper** | fast open-source offline TTS; its voices are the teachers | file 02 |
| **piperlite** | sanoTTS's thinner copy of Piper's decoder | file 05 |
| **QAT / STE** | quantization-aware training / straight-through estimator | file 07 |
| **Quantization** | storing numbers in fewer bits (e.g. int8) | file 07 |
| **Residual block** | `x + small_change(x)`: learns a correction | files 01, 05 |
| **RTF** | real-time factor = compute time ÷ audio length (<1 is faster than real time) | file 07 |
| **Sample / sample rate** | one audio measurement / measurements per second (22,050 Hz here) | file 02 |
| **Schwa deletion** | whether the inherent 'अ' is pronounced; a classic Nepali/Hindi G2P problem | file 08 |
| **Signature (activation signature)** | summaries of the teacher decoder's internal activations used as extra guidance | file 06 |
| **SIMD** | CPU instructions that do many multiplies at once | file 07 |
| **Step** | one training iteration | file 01 |
| **STFT** | Short-Time Fourier Transform: sound → frequencies over time | file 02 |
| **Tensor** | a multi-dimensional array of numbers (PyTorch's word for it) | file 01 |
| **token_context** | the acoustic architecture that encodes phonemes before expanding to frames | file 05 |
| **VITS** | the end-to-end TTS architecture inside every Piper voice | file 02 |
| **w_ceil** | the teacher's integer frame counts per phoneme | files 02, 03 |
| **WASM / WebAssembly** | a way to run C code in the browser at near-native speed | file 07 |
| **z-anchor** | joint-finetune loss keeping the student latent near the teacher's | file 06 |
| **z-mix** | training the decoder partly on the acoustic student's (imperfect) latents | files 03, 04 |

---

## Part B — Every official link, grouped

### Python & NumPy
- Python tutorial — https://docs.python.org/3/tutorial/
- argparse tutorial — https://docs.python.org/3/howto/argparse.html
- subprocess — https://docs.python.org/3/library/subprocess.html
- NumPy for absolute beginners — https://numpy.org/doc/stable/user/absolute_beginners.html
- numpy.repeat — https://numpy.org/doc/stable/reference/generated/numpy.repeat.html

### PyTorch
- Learn the Basics — https://pytorch.org/tutorials/beginner/basics/intro.html
- Autograd — https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
- nn.Conv1d — https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
- nn.ConvTranspose1d — https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
- nn.Embedding — https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
- nn.SiLU — https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
- nn.LeakyReLU — https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
- SmoothL1Loss — https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
- AdamW — https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
- torch.stft — https://pytorch.org/docs/stable/generated/torch.stft.html
- torch.topk — https://pytorch.org/docs/stable/generated/torch.topk.html
- Quantization — https://pytorch.org/docs/stable/quantization.html

### ONNX
- ONNX intro — https://onnx.ai/onnx/intro/
- ONNX Python API — https://onnx.ai/onnx/api/index.html
- ONNX utils (extract_model) — https://onnx.ai/onnx/api/utils.html
- ONNX Runtime (Python) — https://onnxruntime.ai/docs/get-started/with-python.html
- Netron (graph viewer) — https://netron.app/

### TTS: Piper, espeak-ng, voices
- Piper (current) — https://github.com/OHF-Voice/piper1-gpl
- Piper CLI docs — https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/CLI.md
- Piper training docs — https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/TRAINING.md
- Piper (original, archived) — https://github.com/rhasspy/piper
- Piper voice list — https://github.com/rhasspy/piper/blob/master/VOICES.md
- Piper voices on Hugging Face — https://huggingface.co/rhasspy/piper-voices
- Nepali Piper voices — https://huggingface.co/rhasspy/piper-voices/tree/main/ne/ne_NP
- chitwan model card — https://huggingface.co/rhasspy/piper-voices/blob/main/ne/ne_NP/chitwan/medium/MODEL_CARD
- google (Nepali) model card — https://huggingface.co/rhasspy/piper-voices/blob/main/ne/ne_NP/google/medium/MODEL_CARD
- OHF-Voice voice datasets — https://github.com/OHF-Voice/voice-datasets
- espeak-ng — https://github.com/espeak-ng/espeak-ng
- espeak-ng languages — https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md
- IPA chart — https://www.ipachart.com/
- Kokoro-82M (the "heart" voices' teacher) — https://huggingface.co/hexgrad/Kokoro-82M

### Papers
- Knowledge distillation — https://arxiv.org/abs/1503.02531
- VITS — https://arxiv.org/abs/2106.06103
- HiFi-GAN — https://arxiv.org/abs/2010.05646
- FastSpeech 2 (length regulator) — https://arxiv.org/abs/2006.04558
- WaveNet (dilated convolutions) — https://arxiv.org/abs/1609.03499
- ResNet (residual blocks) — https://arxiv.org/abs/1512.03385
- GAN — https://arxiv.org/abs/1406.2661
- LSGAN — https://arxiv.org/abs/1611.04076
- Parallel WaveGAN (multi-resolution STFT loss) — https://arxiv.org/abs/1910.11480
- pix2pix (PatchGAN) — https://arxiv.org/abs/1611.07004
- Quantization whitepaper — https://arxiv.org/abs/1806.08342
- Straight-through estimator — https://arxiv.org/abs/1308.3432

### Evaluation
- SCOREQ — https://github.com/alessandroragano/scoreq
- UTMOS — https://github.com/sarulab-speech/UTMOS22
- DNSMOS — https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS
- Whisper — https://github.com/openai/whisper
- jiwer (WER/CER) — https://github.com/jitsi/jiwer

### Nepali data & STT
- OpenSLR SLR43 (Nepali TTS) — https://www.openslr.org/43/
- OpenSLR SLR54 (Nepali ASR) — https://www.openslr.org/54/
- Mozilla Common Voice — https://commonvoice.mozilla.org/
- Google FLEURS — https://huggingface.co/datasets/google/fleurs
- Montreal Forced Aligner — https://montreal-forced-aligner.readthedocs.io/
- Fine-tune Whisper (HF blog) — https://huggingface.co/blog/fine-tune-whisper
- Fine-tune XLS-R wav2vec2 (HF blog) — https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
- Meta MMS in Transformers — https://huggingface.co/docs/transformers/model_doc/mms
- AI4Bharat — https://ai4bharat.iitm.ac.in/

### Deployment
- WebAssembly — https://webassembly.org/
- Emscripten — https://emscripten.org/docs/getting_started/index.html
- Web Audio API — https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API
- ESP-IDF (ESP32-S3) — https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/
- ESP-NN — https://github.com/espressif/esp-nn
- ESPHome — https://esphome.io/
- GPL FAQ — https://www.gnu.org/licenses/gpl-faq.html

### Courses
- Hugging Face Audio Course — https://huggingface.co/learn/audio-course
- fast.ai Practical Deep Learning — https://course.fast.ai/

### This project
- sanoTTS live demo — https://ampixa.github.io/sanoTTS/
- sanoTTS on Hugging Face — https://huggingface.co/ampixa/sanoTTS
- sanoTTS on npm — https://www.npmjs.com/package/sanotts-web
- sanoTTS-jp (independent Japanese reimplementation of the recipe) — https://github.com/ayutaz/sanoTTS-jp

---

← Back to **[00-START-HERE.md](00-START-HERE.md)**
