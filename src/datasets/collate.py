import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    mix_audios = []
    # mix_specs = []
    mix_audio_paths = []
    # mix_spec_lengths = []
    mix_audio_lengths = []
    s1_audios = []
    # s1_specs = []
    s1_audio_paths = []
    # s1_spec_lengths = []
    s2_audios = []
    # s2_specs = []
    s2_audio_paths = []
    # s2_spec_lengths = []
    

    for cur_dict in dataset_items:
        mix_audios.append(torch.transpose(torch.tensor(cur_dict["mix_audio"]), 0, 1))
        # mix_spec_lengths.append(cur_dict["mix_spectrogram"].shape[2])
        mix_audio_lengths.append(cur_dict["mix_audio"].shape[1])
        # mix_specs.append(torch.transpose(torch.tensor(cur_dict["mix_spectrogram"]), 0, 2))
        mix_audio_paths.append(cur_dict["mix_audio_path"])  # not to pad
        s1_audios.append(torch.transpose(torch.tensor(cur_dict["s1_audio"]), 0, 1))
        # s1_spec_lengths.append(cur_dict["s1_spectrogram"].shape[2])
        # s1_specs.append(torch.transpose(torch.tensor(cur_dict["s1_spectrogram"]), 0, 2))
        s1_audio_paths.append(cur_dict["s1_audio_path"])  # not to pad
        s2_audios.append(torch.transpose(torch.tensor(cur_dict["s2_audio"]), 0, 1))
        # s2_spec_lengths.append(cur_dict["s2_spectrogram"].shape[2])
        # s2_specs.append(torch.transpose(torch.tensor(cur_dict["s2_spectrogram"]), 0, 2))
        s2_audio_paths.append(cur_dict["s2_audio_path"])  # not to pad

    result = {}

    result["mix_audio_path"] = mix_audio_paths
    # result["mix_spectrogram_length"] = torch.tensor(mix_spec_lengths)
    # result["mix_audio_length"] = torch.tensor(mix_spec_lengths)
    result["mix_audio_length"] = torch.tensor(mix_audio_lengths)
    result["mix_audio"] = torch.nn.utils.rnn.pad_sequence(mix_audios, batch_first=True)
    # result["mix_spectrogram"] = torch.nn.utils.rnn.pad_sequence(mix_specs, batch_first=True)
    result["mix_audio"] = torch.transpose(result["mix_audio"], 1, 2).squeeze(1)
    # result["mix_spectrogram"] = torch.transpose(result["mix_spectrogram"], 1, 3).squeeze(1)

    result["s1_audio_path"] = s1_audio_paths
    # result["s1_spectrogram_length"] = torch.tensor(s1_spec_lengths)
    # result["s1_audio_length"] = torch.tensor(s1_spec_lengths)
    result["s1_audio"] = torch.nn.utils.rnn.pad_sequence(s1_audios, batch_first=True)
    # result["s1_spectrogram"] = torch.nn.utils.rnn.pad_sequence(s1_specs, batch_first=True)
    result["s1_audio"] = torch.transpose(result["s1_audio"], 1, 2).squeeze(1)
    # result["s1_spectrogram"] = torch.transpose(result["s1_spectrogram"], 1, 3).squeeze(1)

    result["s2_audio_path"] = s2_audio_paths
    # result["s2_spectrogram_length"] = torch.tensor(s2_spec_lengths)
    # result["s2_audio_length"] = torch.tensor(s2_spec_lengths)
    result["s2_audio"] = torch.nn.utils.rnn.pad_sequence(s2_audios, batch_first=True)
    # result["s2_spectrogram"] = torch.nn.utils.rnn.pad_sequence(s2_specs, batch_first=True)
    result["s2_audio"] = torch.transpose(result["s2_audio"], 1, 2).squeeze(1)
    # result["s2_spectrogram"] = torch.transpose(result["s2_spectrogram"], 1, 3).squeeze(1)

    return result

