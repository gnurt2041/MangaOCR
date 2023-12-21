import fire
import wandb
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
import sys
sys.path.append("..")
from MangaOCR.env import TRAIN_ROOT, DATA_ROOT
from MangaOCR.data.dataset import Manga109, train_val_split
from get_model import get_model
from metrics import Metrics

def run(
        run_name='debug',
        encoder_name='facebook/deit-tiny-patch16-224',
        decoder_name='cl-tohoku/bert-base-japanese-char-v2',
        output = TRAIN_ROOT,
        max_len=300,
        num_decoder_layers=2,
        batch_size=64,
        num_epochs=8,
        fp16=True,
):
    wandb.login()

    model, processor = get_model(encoder_name, decoder_name, max_len, num_decoder_layers)

    # keep package 0 for validation
    # train_dataset = Manga109(processor, 'train', max_len, augment=True, skip_packages=[0])
    # eval_dataset = Manga109(processor, 'test', max_len, augment=False, skip_packages=range(1, 9999))

    dataset = Manga109(DATA_ROOT ,'./text_img.csv' , processor, augment = True, max_length = 300)
    train_dataset, val_datset, test_datset = train_val_split(dataset)

    metrics = Metrics(processor)

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy='steps',
        save_strategy='steps',
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=fp16,
        fp16_full_eval=fp16,
        dataloader_num_workers=2,
        output_dir=output,
        logging_steps=10,
        save_steps=1000,
        eval_steps=200,
        num_train_epochs=num_epochs,
        run_name=run_name
    )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.image_processor,
        args=training_args,
        compute_metrics=metrics.compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_datset,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model("./translation-output")

    wandb.finish()


if __name__ == '__main__':
    fire.Fire(run)
