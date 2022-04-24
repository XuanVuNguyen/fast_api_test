import torch
from model import Model, TextPreprocessing, id2label

class Predictor:
    def __init__(
        self,
        model_checkpoint,
        vocab_path):
        self.model = Model.load_from_checkpoint(model_checkpoint)
        self.model.eval()
        self.text_preprocessing = TextPreprocessing(
            self.model.config.max_seq_len,
            vocab_save_file=vocab_path
            )
        self.id2label = id2label
    
    def predict(self, text: str):
        outputs = self.text_preprocessing.text2tensor(text)
        outputs = self.model(outputs)
        max_prob, label_id = torch.max(outputs, dim=-1)
        label = self.id2label[label_id.item()]
        return label, max_prob.item()

predictor = Predictor("lightning_logs/version_0/checkpoints/epoch=19-step=1680.ckpt", "vocab.pth")