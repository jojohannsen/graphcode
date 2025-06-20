from llm_cache import make_llm

def node_chat_model():
    return make_llm("{llm_provider}", "{llm_model}", "{cache_type}")