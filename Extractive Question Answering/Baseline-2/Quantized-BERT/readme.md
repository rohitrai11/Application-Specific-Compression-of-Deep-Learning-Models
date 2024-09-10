Fine tuned BERT-base model from Baseline-1 is quantized using pytorch built in quantization library:

quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

No fine tuning has been done in this case. Just the quantized model is evaluated on the dev set.

Run the evaluation script using the command: bash eval_quantized_bert_squad.sh
