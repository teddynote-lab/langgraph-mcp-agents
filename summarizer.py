# real_summarizer.py
import sys
import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

# 미리 학습된 한국어 요약 모델과 토크나이저를 불러옵니다.
# 이 모델은 Hugging Face Hub에서 자동으로 다운로드됩니다.
model_name = 'gogamza/kobart-summarization'
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def summarize(text):
    """
    주어진 텍스트를 AI 모델을 사용하여 요약하는 함수.
    """
    # 1. 입력 텍스트를 모델이 이해할 수 있는 숫자 형태(토큰)로 변환합니다.
    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")

    # 2. 모델을 사용하여 요약 텍스트를 생성합니다.
    #    - max_length: 요약문의 최대 길이
    #    - num_beams: 더 좋은 문장을 찾기 위한 탐색 빔의 수
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=256, early_stopping=True)

    # 3. 생성된 숫자 형태의 요약문을 다시 사람이 읽을 수 있는 텍스트로 변환합니다.
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary_text

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 커맨드 라인에서 두 번째 인자(요약할 텍스트)를 가져옵니다.
        input_text = sys.argv[1]
        summary = summarize(input_text)
        print(summary)
    else:
        print("요약할 텍스트를 입력하세요.")