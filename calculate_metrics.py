import numpy as np
from pathlib import Path
from src.metrics.utils import calc_si_sdri, calc_si_snri
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
    si_snris = []

    mix_path = config.paths.mix_path
    s1_estimated = config.paths.s1_estimated_path
    s2_estimated = config.paths.s2_estimated_path
    s1_target = config.paths.s1_target_path
    s2_target = config.paths.s2_target_path
    
    if not s1_estimated or not Path(s1_estimated).exists():
        print("Invalid s1 estimated path")
        return
    
    for estimated in Path(s1_estimated).iterdir():
        if estimated.suffix == ".wav":
            if mix_path and s2_estimated and Path(mix_path).exists() and Path(s2_estimated).exists():
                s1_est = Path(s1_estimated) / (estimated.stem + estimated.suffix)
                s2_est = Path(s2_estimated) / (estimated.stem + estimated.suffix)
                mix = Path(mix_path) / (estimated.stem + estimated.suffix)

                s1_predicted = load_audio(s1_est, config.vars.target_sr)
                s2_predicted = load_audio(s2_est, config.vars.target_sr)
                mix_audio = load_audio(mix, config.vars.target_sr)
                s1_audio = None
                s2_audio = None
                sdr_s1_s1 = None
                sdr_s1_s2 = None
                sdr_s2_s1 = None
                sdr_s2_s2 = None

                snr_s1_s1 = None
                snr_s1_s2 = None
                snr_s2_s1 = None
                snr_s2_s2 = None

                if s1_target and Path(s1_target).exists():
                    s1_true = Path(s1_target) / (estimated.stem + estimated.suffix)
                    s1_audio = load_audio(s1_true, config.vars.target_sr)

                    sdr_s1_s1 = calc_si_sdri(s1_audio, s1_predicted, mix_audio)
                    sdr_s1_s2 = calc_si_sdri(s1_audio, s2_predicted, mix_audio)

                    snr_s1_s1 = calc_si_snri(s1_audio, s1_predicted, mix_audio)
                    snr_s1_s2 = calc_si_snri(s1_audio, s2_predicted, mix_audio)

                if s2_target and Path(s2_target).exists():
                    s2_true = Path(s2_target) / (estimated.stem + estimated.suffix)
                    s2_audio = load_audio(s2_true, config.vars.target_sr)

                    sdr_s2_s1 = calc_si_sdri(s2_audio, s1_predicted, mix_audio)
                    sdr_s2_s2 = calc_si_sdri(s2_audio, s2_predicted, mix_audio)

                    snr_s2_s1 = calc_si_snri(s2_audio, s1_predicted, mix_audio)
                    snr_s2_s2 = calc_si_snri(s2_audio, s2_predicted, mix_audio)

                if sdr_s1_s1 is not None and sdr_s2_s2 is not None:
                    si_sdris.append(max((sdr_s1_s1 + sdr_s2_s2) / 2, (sdr_s1_s2 + sdr_s2_s1) / 2))
                    si_snris.append(max((snr_s1_s1 + snr_s2_s2) / 2, (snr_s1_s2 + snr_s2_s1) / 2))
                elif sdr_s1_s1 is not None:
                    si_sdris.append(max(sdr_s1_s1, sdr_s1_s2))
                    si_snris.append(max(snr_s1_s1, snr_s1_s2))
                elif sdr_s2_s2 is not None:
                    si_sdris.append(max(sdr_s2_s1, sdr_s2_s2))
                    si_snris.append(max(snr_s2_s1, snr_s2_s2))
                else:
                    si_sdris.append(0)
                    si_snris.append(0)

            else:
                print("Invalid path")
                return


    print("SI-SDRi metric for given directory:", np.mean(si_sdris))
    print("SI-SNRi metric for given directory:", np.mean(si_snris))



if __name__ == "__main__":
    main()
