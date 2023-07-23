import hydra
from src.models.gpt4all_model import MyGPT4ALL

@hydra.main(config_path='./configs', config_name='config')
def main(cfg):

    # instantiate the model and populate the arguments using hydra

    chat_model = MyGPT4ALL(
        model_folder_path=cfg.model.gpt4all_model.gpt4all_model_folder_path,
        model_name=cfg.model.gpt4all_model.gpt4all_model_name,
        allow_download=cfg.model.gpt4all_model.gpt4all_allow_downloading,
        allow_streaming=cfg.model.gpt4all_model.gpt4all_allow_streaming,
        
    )

    while True:
        query = input('Enter your Query: ')
        if query == 'exit':
            break
        # use hydra to fill the **kwargs
        response = chat_model(
            query,
            n_predict=cfg.model.gpt4all_model.gpt4all_n_predict,
            temp=cfg.model.gpt4all_model.gpt4all_temperature,
            top_p=cfg.model.gpt4all_model.gpt4all_top_p,
            top_k=cfg.model.gpt4all_model.gpt4all_top_k,
            n_batch=cfg.model.gpt4all_model.gpt4all_n_batch,
            repeat_last_n=cfg.model.gpt4all_model.gpt4all_repeat_last_n,
            repeat_penalty=cfg.model.gpt4all_model.gpt4all_penalty,
            max_tokens=cfg.model.gpt4all_model.gpt4all_max_tokens,
        )
        print()

if __name__ == '__main__':
    main()