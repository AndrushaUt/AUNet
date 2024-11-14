import numpy as np
from pathlib import Path
from src.metrics.utils import calc_si_sdri
import hydra
import torchaudio


def load_audio(path, target_sr):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

@hydra.main(version_base=None, config_path="src/configs", config_name="metrics")
def main(config):
    si_sdris = []

    s1_estimated = config.paths.s1_estimated_path
    s2_estimated = config.paths.s2_estimated_path
    s1_target = config.paths.s1_target_path
    s2_target = config.paths.s2_target_path
    mix_path = config.paths.mix_path

    if not s1_estimated or not Path(s1_estimated).exists():
        print("Invalid s1 estimated path")
        return
    
    for estimated in Path(s1_estimated).iterdir():
        if estimated.suffix == ".wav":
            if mix_path and s2_estimated and s1_target and s2_target and \
                Path(mix_path).exists() and Path(s2_estimated).exists() and \
                    Path(s1_target).exists() and Path(s2_target).exists():
            
                s1_est = Path(s1_estimated) / (estimated.stem + estimated.suffix)
                s2_est = Path(s2_estimated) / (estimated.stem + estimated.suffix)
                s1_true = Path(s1_target) / (estimated.stem + estimated.suffix)
                s2_true = Path(s2_target) / (estimated.stem + estimated.suffix)
                mix = Path(mix_path) / (estimated.stem + estimated.suffix)

                s1_audio = load_audio(s1_true, config.vars.target_sr)
                s2_audio = load_audio(s2_true, config.vars.target_sr)
                s1_predicted = load_audio(s1_est, config.vars.target_sr)
                s2_predicted = load_audio(s2_est, config.vars.target_sr)
                mix_audio = load_audio(mix, config.vars.target_sr)

                s1_s1 = calc_si_sdri(s1_audio, s1_predicted, mix_audio)
                s1_s2 = calc_si_sdri(s1_audio, s2_predicted, mix_audio)
                s2_s1 = calc_si_sdri(s2_audio, s1_predicted, mix_audio)
                s2_s2 = calc_si_sdri(s2_audio, s2_predicted, mix_audio)

                si_sdris.append(max((s1_s1 + s2_s2) / 2, (s1_s2 + s2_s1) / 2))
            else:
                print("Invalid path")
        else:
            print("Invalid format")

    print("SI-SDRi metric for given directory:", np.mean(si_sdris))


if __name__ == "__main__":
    main()
