import torchaudio

from pathlib import Path

from src.datasets.base_dataset import BaseDataset


class CustomDatasetAudioOnly(BaseDataset):
    def __init__(self, mix_audio_dir, s1_audio_dir=None, s2_audio_dir=None, *args, **kwargs):
        data = []
        for mix_path in Path(mix_audio_dir).iterdir():
            entry = {}
            if mix_path.suffix in [".mp3", ".wav", ".flac"]:
                entry["mix_path"] = str(mix_path)
                entry["mix_audio_length"] = self._calculate_length(mix_path)
                if s1_audio_dir and s2_audio_dir and Path(s1_audio_dir).exists() and Path(s2_audio_dir).exists():
                    s1_path = Path(s1_audio_dir) / (mix_path.stem + mix_path.suffix)
                    if s1_path.suffix == mix_path.suffix:
                        entry["s1_path"] = str(s1_path)
                        entry["s1_audio_length"] = self._calculate_length(s1_path)

                    s2_path = Path(s2_audio_dir) / (mix_path.stem + mix_path.suffix)
                    if s2_path.suffix == mix_path.suffix:
                        entry["s2_path"] = str(s2_path)
                        entry["s2_audio_length"] = self._calculate_length(s2_path)
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)
    
    def _calculate_length(self, audio_path: Path) -> float:
        audio_info = torchaudio.info(str(audio_path))
        return audio_info.num_frames / audio_info.sample_rate
