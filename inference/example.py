from masking_api import mask_text_crf, get_entity_locations_crf

def main():
    sample_text = "철수야 밥 먹었어?"
    masked_text = mask_text_crf(sample_text)
    locations = get_entity_locations_crf(sample_text)
    
    print("입력 텍스트(Korean Input):", sample_text)
    print("마스킹 결과(Masked Output):", masked_text)
    print("위치정보(Location Info):", locations)

if __name__ == "__main__":
    main()
