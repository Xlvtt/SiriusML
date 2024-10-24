import datasets
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizerFast


MAX_LEN = 256
DOC_STRIDE = 64
MAX_ANS_LEN = 256
MIN_ANS_LEN = 2
tokenizer = BertTokenizerFast("../vocab.txt", do_lower_case=False, clean_up_tokenization_spaces=True, padding_side="right")


def prepare_test_data(examples: dict):
    examples["question"] = [q.lstrip() for q in examples["question"]]

    tokenized_samples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=MAX_LEN,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    id_list = []
    masked_offset_mapping = []
    overflow_to_sample_mapping = tokenized_samples.pop("overflow_to_sample_mapping")
    for i in range(len(tokenized_samples["input_ids"])):
        sample_index = overflow_to_sample_mapping[i]
        id_list.append(examples["id"][sample_index])

        sequence_ids = tokenized_samples.sequence_ids(i)
        offsets = tokenized_samples["offset_mapping"][i]
        masked_offset_mapping.append([offset if sequence_ids[index] == 1 else None for index, offset in enumerate(offsets)])

    tokenized_samples["id"] = id_list
    tokenized_samples["offset_mapping"] = masked_offset_mapping
    return tokenized_samples


def inference(
        model, data: dict | datasets.Dataset,
        n_best_size=20, max_answer_len=MAX_ANS_LEN, min_answer_len=MIN_ANS_LEN
    ):
    """
    :param model: моделька
    :param data: словарь формата теста (поля question, context и id)
    :param n_best_size: сколько лучших значений логитов используется для поиска ответа
    :param max_answer_len: не рассматриваем ответы с большей длиной даже если у них высокий скор
    :param min_answer_len: аналогично не рассматриваем меньшую длину
    :return: ответ на вопрос
    """
    if isinstance(data, datasets.Dataset):
        tokenized_data = data.map(prepare_test_data, batched=True, remove_columns=data.column_names)
    else:
        tokenized_data = prepare_test_data(data)

    preds = model(
        input_ids=torch.tensor(tokenized_data["input_ids"], dtype=torch.int64),
        attention_mask=torch.tensor(tokenized_data["attention_mask"], dtype=torch.int64)
    )
    get_pieces_index = {}  # словарь id существующего текста: индексы всех его кусков после токенизации
    for i, sample_index in enumerate(tokenized_data["id"]):
        if sample_index not in get_pieces_index:
            get_pieces_index[sample_index] = []
        get_pieces_index[sample_index].append(i)

    total_answers = []
    for sample_index, sample in enumerate(tqdm(data["id"])):
        context = data["context"][sample_index]  # текущий контекст

        valid_answers = []  # ответы для одного контекста
        for piece_index in get_pieces_index[
            sample]:  # перебираем все куски одного контекста, собираем ответы для каждого
            start_logits = preds.start_logits[piece_index].detach().numpy()
            end_logits = preds.end_logits[piece_index].detach().numpy()
            offset_mapping = tokenized_data["offset_mapping"][piece_index]

            best_start_tokens_ids = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            best_end_tokens_ids = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()

            for start_token in best_start_tokens_ids:
                for end_token in best_end_tokens_ids:
                    if offset_mapping[start_token] is None or offset_mapping[end_token] is None:
                        continue
                    if (
                            start_token > end_token or
                            end_token - start_token + 1 > max_answer_len or
                            end_token - start_token + 1 < min_answer_len
                    ):
                        continue
                    valid_answers.append({
                        "score": start_logits[start_token].item() + end_logits[end_token].item(),
                        "text": context[offset_mapping[start_token][0]: offset_mapping[end_token][1]]
                    })
            if not valid_answers:
                valid_answers.append({"score": 0.0, "text": ""})

        total_answers.append(
            max(valid_answers, key=lambda x: x["score"]))  # для всего контекста берем лучший по всем кусочкам

    return total_answers
