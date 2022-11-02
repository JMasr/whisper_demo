import jiwer
import pandas as pd

from dataset import *
from tqdm import tqdm
from whisper.normalizers import BasicTextNormalizer


print("Whisper demo")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base")

print(f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
      f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters.")

dataset = StandarDataset(audio_path='data/tvgal_wav', transcription_path='data/tvgal_txt', language='gl')
loader = torch.utils.data.DataLoader(dataset, batch_size=16)

hypotheses, references = [], []
for mel, text, audio_path in tqdm(loader):

    # detect the spoken language
    _, probs = model.detect_language(mel)
    lang_list = sorted(probs[0], key=probs[0].get, reverse=True)
    lang = max(probs[0], key=probs[0].get)
    print(f"Detected language: {lang}")
    print(f"Detected languages: {lang_list[:5]}")

    # Decoding
    options_deco = whisper.DecodingOptions(language=dataset.language, without_timestamps=False)
    results = model.decode(mel, options_deco)
    hypotheses.extend([result.text for result in results])
    references.extend(text)
    print('Decoding DONE!')

data = pd.DataFrame(dict(reference=references, transcription=hypotheses))
# Normalize texts
normalizer = BasicTextNormalizer()
data["reference_clean"] = [normalizer(text) for text in data["reference"]]
data["hypothesis_clean"] = [normalizer(text) for text in data["transcription"]]
# Save data
data.to_csv('results/whisper_tvgal.csv')
# Calculate WER
wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))
print(f"WER: {wer * 100:.2f} %")

