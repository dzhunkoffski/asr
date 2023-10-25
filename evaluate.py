import argparse
import json
from hw_asr.metric.utils import calc_wer, calc_cer
from tqdm.auto import tqdm

def main(prediction_path: str):
    wer_argmax = 0
    cer_argmax = 0
    wer_lmbeam = 0
    cer_lmbeam = 0
    wer_beam = 0
    cer_beam = 0
    diy_wer_beam = 0
    diy_cer_beam = 0
    with open(prediction_path, 'rb') as fd:
        predictions = json.load(fd)
    for item in tqdm(predictions):
        target_text = item['ground_truth']
        predicted_argmax = item['pred_text_argmax']
        predicted_beam = item['pred_text_beam_search_without_lm'][0][0]
        predicted_lmbeam = item['pred_text_beam_search_with_lm'][0][0]
        predicted_diybeam = item['pred_text_diy_beamsearch'][0]

        wer_argmax += calc_wer(target_text, predicted_argmax)
        cer_argmax += calc_cer(target_text, predicted_argmax)

        wer_beam += calc_wer(target_text, predicted_beam)
        cer_beam += calc_cer(target_text, predicted_beam)

        wer_lmbeam += calc_wer(target_text, predicted_lmbeam)
        cer_lmbeam += calc_cer(target_text, predicted_lmbeam)

        diy_wer_beam += calc_wer(target_text, predicted_diybeam)
        diy_cer_beam += calc_cer(target_text, predicted_diybeam)
    
    wer_argmax /= len(predictions)
    cer_argmax /= len(predictions)
    wer_beam /= len(predictions)
    cer_beam /= len(predictions)
    wer_lmbeam /= len(predictions)
    cer_lmbeam /= len(predictions)
    diy_wer_beam /= len(predictions)
    diy_cer_beam /= len(predictions)

    print(f'Argmax WER: {wer_argmax}')
    print(f'Argmax CER: {cer_argmax}')
    print(f'BeamSearch WER: {wer_beam}')
    print(f'BeamSearch CER: {cer_beam}')
    print(f'LM BeamSearch WER: {wer_lmbeam}')
    print(f'LM BeamSearch CER: {cer_lmbeam}')
    print(f'DIY beam WER: {diy_wer_beam}')
    print(f'DIY beam CER: {diy_cer_beam}')


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-p",
        "--predictions",
        default=None,
        type=str,
        help="predictions json filepath"
    )
    args = args.parse_args()
    main(args.predictions)