import gdown
import hydra
import os
from pathlib import Path

ROOT_PATH = Path(__file__).absolute().resolve().parent

@hydra.main(version_base=None, config_path="src/configs", config_name="download")
def main(config):
    link = config.vars.link
    save_path = ROOT_PATH / "saved" / config.vars.to_save
    save_path.mkdir(exist_ok=True, parents=True)
    
    gdown.download(link, str(save_path) + "/best_model.pth", fuzzy=True)



if __name__ == "__main__":
    main()

    # if not os.path.exists('saved/model'):
    #     os.mkdir('saved/model')
    # gdown.download("https://drive.google.com/file/d/13V_IdQI_JCwzngyqAq616E4ckjcFRg7E/view?usp=sharing", "saved/model/best_model.pth")
