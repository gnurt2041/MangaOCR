from transformers import AutoConfig, AutoModelForCausalLM, AutoModel, TrOCRProcessor, VisionEncoderDecoderModel, \
    AutoFeatureExtractor, AutoTokenizer, VisionEncoderDecoderConfig

class TrOCRProcessorCustom(TrOCRProcessor):
    def __init__(self, feature_extractor, tokenizer):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.current_processor = self.feature_extractor

def get_processor(encoder_name, decoder_name):
    feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    processor = TrOCRProcessorCustom(feature_extractor, tokenizer)
    return processor